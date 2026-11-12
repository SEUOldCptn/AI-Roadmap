"""AI-Roadmap: Continuous Batching 调度器学生练习（CPU 可运行）.

目标：理解动态加入/退出请求，以及每条请求独立 KV Cache 的生命周期。

放置位置：03-transformer/10-continuous-batching-exercise.py
运行：python 03-transformer/10-continuous-batching-exercise.py
可选（运行 MiniGPT）：python 03-transformer/10-continuous-batching-exercise.py --real

注意：这是 iteration-level 调度器教学模拟。为避免引入变长批量 Attention，
本脚本的 engine 对每条请求分别调用，不等同于一次 GPU batched forward，
不能用其运行时间证明真实 continuous batching 的加速比。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    arrival_step: int
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)
    state: str = "waiting"  # waiting -> prefill -> decode -> finished
    first_token_step: int | None = None
    finished_step: int | None = None
    past_kvs: Any = None
    cache_len: int = 0  # FakeEngine 用于验证缓存长度，仅用于模拟

    def __post_init__(self):
        if not self.request_id or not self.prompt_ids:
            raise ValueError("request_id 和 prompt_ids 不允许为空")
        if self.arrival_step < 0 or self.max_new_tokens < 1:
            raise ValueError("arrival_step >= 0, max_new_tokens >= 1")

    @property
    def ttft_steps(self) -> int | None:
        if self.first_token_step is None:
            return None
        return self.first_token_step - self.arrival_step + 1

    @property
    def latency_steps(self) -> int | None:
        if self.finished_step is None:
            return None
        return self.finished_step - self.arrival_step + 1


class FakeEngine:
    """每次调用产出一个确定的 token，便于只测试调度逻辑。"""

    def prefill(self, request: Request) -> int:
        request.cache_len = len(request.prompt_ids)
        return (sum(request.prompt_ids) + 7) % 101

    def decode(self, request: Request) -> int:
        if not request.generated:
            raise ValueError("decode 前必须完成 prefill")
        # 输入上一轮新生成的 token，因此它现在也进入缓存。
        request.cache_len += 1
        return (request.generated[-1] + 1) % 101


class MiniGPTEngine:
    """进阶练习：直接复用上一课的 KV Cache MiniGPT。非真正批量推理。"""

    def __init__(self):
        import torch
        from kv_cache_exercise import MiniGPT  # 两个脚本放在同一目录

        self.torch = torch
        torch.manual_seed(7)
        self.model = MiniGPT(
            vocab_size=101, context_size=64,
            d_model=32, num_heads=4, num_layers=2,
        ).eval()

    def prefill(self, request: Request) -> int:
        if len(request.prompt_ids) + request.max_new_tokens > self.model.context_size:
            raise ValueError("prompt + max_new_tokens 超出模型 context_size")
        ids = self.torch.tensor([request.prompt_ids], dtype=self.torch.long)
        with self.torch.inference_mode():
            logits, request.past_kvs = self.model(ids, use_cache=True)
            return int(logits[0, -1].argmax().item())

    def decode(self, request: Request) -> int:
        last_id = self.torch.tensor([[request.generated[-1]]], dtype=self.torch.long)
        with self.torch.inference_mode():
            logits, request.past_kvs = self.model(
                last_id, past_kvs=request.past_kvs, use_cache=True
            )
            return int(logits[0, -1].argmax().item())


@dataclass
class StepRecord:
    step: int
    admitted: list[str]
    emitted: list[str]
    finished: list[str]
    active_after: list[str]


class Scheduler:
    def __init__(self, requests: list[Request], engine, *, max_active=2, mode="continuous"):
        if max_active < 1 or mode not in {"continuous", "static"}:
            raise ValueError("max_active >= 1 且 mode 为 continuous/static")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("request_id 必须唯一")
        self.pending = sorted(requests, key=lambda r: (r.arrival_step, r.request_id))
        self.all_requests = list(self.pending)
        self.active: dict[str, Request] = {}
        self.max_active = max_active
        self.mode = mode
        self.engine = engine
        self.history: list[StepRecord] = []

    def admit(self, step: int) -> list[str]:
        """TODO 1：把到达且容量允许的请求从 pending 加入 active。
        continuous: 任意空槽都可以招入新请求。
        static: 只在 active 完全为空时开启下一批，批次中途不补位。
        提示：pending 已按 arrival_step 排序。
        入场时将 request.state 从 waiting 设置为 prefill。
        返回本轮 admitted 的 request_id 列表。
        """
        output = []
        if self.mode == "static" and self.active:
            return output
        while self.pending and len(self.active) < self.max_active:
        # 只要等待队列不为空，并且执行批次还有位置，就尝试接纳新的请求
            if self.pending[0].arrival_step > step:
                # 当前的step还没有处理到队首的request，因此不接纳新的请求
                break
            request = self.pending.pop(0)
            request.state = "prefill"
            self.active[request.request_id] = request
            output.append(request.request_id)
        return output

    def execute_step(self, step: int) -> tuple[list[str], list[str]]:
        """TODO 2：每个 active 请求在当前迭代至多产生一个 token。

        state == 'prefill': engine.prefill(request)，随后进入 decode。
        state == 'decode':  engine.decode(request)，保持 decode。
        每次产生 token：追加 request.generated。
        如果首次产生 token：设置 first_token_step = step。
        达到 max_new_tokens：state='finished'、设置 finished_step，
                         并立即从 active 中移除（遍历时不要原地修改 dict）。
        返回 (emitted_ids, finished_ids)。
        """
        emitted_ids = []
        finished_ids = []
        for request_id, request in self.active.items():
            if request.state == "prefill":
                token = self.engine.prefill(request)
                request.state = "decode"
            elif request.state == "decode":
                token = self.engine.decode(request)
            else:
                raise ValueError(f"Unknown Request State: {request.state}")
            request.generated.append(token)
            if len(request.generated) == 1:
                request.first_token_step = step
            emitted_ids.append(request.request_id)
            if len(request.generated) >= request.max_new_tokens:
                request.state = "finished"
                request.finished_step = step
                finished_ids.append(request.request_id)
        for request_id in finished_ids:
            self.active.pop(request_id)
        return emitted_ids, finished_ids

    def run(self) -> list[StepRecord]:
        """TODO 3：实现离散时间步调度循环。

        从 step=0 开始；只要 pending 或 active 非空就继续。
        每轮：admit -> execute_step -> 记录 StepRecord -> step += 1。
        允许某轮没有可运行请求（等待之后的 arrival_step）。
        注意：本模拟在同一个时间步内逻辑上处理每个 active 请求一次，
        不模拟真实 GPU kernel 时长，也不在此处实现并行执行。
        """
        step = 0
        while self.pending or self.active:
            admitted = self.admit(step)
            emitted_ids, finished_ids = self.execute_step(step)
            record = StepRecord(step=step, admitted=admitted, emitted=emitted_ids, finished=finished_ids, active_after=sorted(self.active))
            self.history.append(record)
            step += 1
        return list(self.history)


    @property
    def makespan_steps(self) -> int:
        return len(self.history)


# --------------------------- 以下为验收测试 ---------------------------

def sample_requests() -> list[Request]:
    return [
        Request("A", [1, 2, 3], arrival_step=0, max_new_tokens=4),
        Request("B", [4, 5], arrival_step=0, max_new_tokens=1),
        Request("C", [6], arrival_step=1, max_new_tokens=2),
    ]


def print_trace(sched: Scheduler):
    print(f"\n[{sched.mode.upper()}] 迭代日志")
    print("step | admitted | emitted  | finished | active after")
    for h in sched.history:
        print(f"{h.step:>4} | {','.join(h.admitted):<8} | "
              f"{','.join(h.emitted):<8} | {','.join(h.finished):<8} | "
              f"{','.join(h.active_after)}")
    for r in sched.all_requests:
        print(f"  {r.request_id}: output={r.generated}, TTFT={r.ttft_steps} steps, "
              f"latency={r.latency_steps} steps")
    print(f"makespan={sched.makespan_steps} steps")


def test_scheduler():
    continuous = Scheduler(sample_requests(), FakeEngine(), max_active=2, mode="continuous")
    static = Scheduler(sample_requests(), FakeEngine(), max_active=2, mode="static")
    continuous.run()
    static.run()

    assert continuous.makespan_steps == 4, "动态补位应在 4 步内完成"
    assert static.makespan_steps == 6, "静态批次应在 6 步内完成"
    assert continuous.history[1].admitted == ["C"], "B 完成后应立即加入 C"
    for s in (continuous, static):
        assert all(r.state == "finished" for r in s.all_requests)
        assert all(len(r.generated) == r.max_new_tokens for r in s.all_requests)
        assert all(len(h.emitted) <= 2 for h in s.history)
    out_c = {r.request_id: r.generated for r in continuous.all_requests}
    out_s = {r.request_id: r.generated for r in static.all_requests}
    assert out_c == out_s, "调度方式不应改变 MockEngine 的 token 序列"
    req_c = {r.request_id: r for r in continuous.all_requests}
    assert req_c["C"].ttft_steps == 1
    assert req_c["A"].cache_len == len(req_c["A"].prompt_ids) + 3
    print("[PASS] 动态加入/结束、输出等价、缓存长度及调度时间测试")
    print_trace(continuous)
    print_trace(static)


def test_real_minigpt():
    print("\n[可选] 接入上一课 MiniGPT + KV Cache")
    import torch
    torch.set_num_threads(1)
    reqs = sample_requests()
    scheduler = Scheduler(reqs, MiniGPTEngine(), max_active=2, mode="continuous")
    scheduler.run()
    assert all(len(r.generated) == r.max_new_tokens for r in reqs)
    # 验证 MiniGPT 分层缓存长度：最后一个新 token 尚未输入模型，无需缓存。
    for r in reqs:
        length = r.past_kvs[0][0].size(-2)
        assert length == len(r.prompt_ids) + r.max_new_tokens - 1
    print("[PASS] 每请求独立 KV Cache、分层缓存长度")
    print_trace(scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="同时运行真实 MiniGPT 适配测试")
    args = parser.parse_args()
    test_scheduler()
    if args.real:
        test_real_minigpt()
