"""AI-Roadmap 2.0 | Phase A, Week 02 | Training, Validation & Overfitting.

学生练习：核心 TODO 需要独立完成。本文件不读取或依赖 11/12 号 AI 生成脚本。
前置：你写过的 04-train-evolution/01-Dataset-Engineering.py，以及
      03-transformer/08-mini-gpt.py（作为模型结构使用）。

在 AI-Roadmap 仓库根目录运行：
    python -m pip install torch matplotlib
    python "04-train-evolution/02-training-validation-overfitting.py" --test-only
    python "04-train-evolution/02-training-validation-overfitting.py" --steps 250

注意：只有 20 篇短文档的教学数据，不能用于真实 LLM 泛化能力结论。
本课使用预定义 ASCII 字符词表，避免验证集未知 token 干扰正则化对比。
因此本课的 PPL 是 char-level PPL，不能直接与 BPE-based LLM PPL 比较。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "03-transformer" / "08-mini-gpt.py"
DATASET_PATH = REPO_ROOT / "04-train-evolution" / "01-Dataset-Engineering.py"
OUTPUT_DIR = REPO_ROOT / "04-train-evolution" / "week02-results"

# 预定义固定词表：本课重点不是再学一次 Tokenizer，而是研究 train/val。
# 所有训练与验证文档共享这个词表，且从未利用验证文档更新模型或拟合词表。
VOCAB = "".join(chr(code) for code in range(32, 127)) + "\n"
CHAR_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"缺少前置课程文件：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_documents():
    """沿用自己 Week 01 实现的文档级数据切分，不跨 split，也不跨文档采样。"""
    ds = load_module(DATASET_PATH, "student_dataset_week01")
    splits = ds.split_documents(ds.DOCUMENTS, seed=SEED)
    keys = ("train", "val", "test")
    sets = {k: set(splits[k]) for k in keys}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert set.union(*sets.values()) == set(range(len(ds.DOCUMENTS)))
    result = {}
    for part in keys:
        texts = [ds.DOCUMENTS[i] + "\n" for i in splits[part]]
        for sentence in texts:
            unsupported = set(sentence) - set(CHAR_TO_ID)
            if unsupported:
                raise ValueError(f"固定 ASCII 词表未覆盖字符：{unsupported}")
        result[part] = [
            torch.tensor([CHAR_TO_ID[ch] for ch in doc], dtype=torch.long)
            for doc in texts
        ]
        print(f"{part:>5} | documents={len(texts):2d} | "
              f"characters={sum(len(x) for x in result[part])}")
    return result


# ====================== 核心学生作业：5 个 TODO ======================


def sample_batch(documents: list[torch.Tensor], batch_size: int, context_size: int):
    """TODO 1：从一组独立文档中随机采样，返回 x、y，shape 均为 (B,T)。

    要求：
    - 每条样本都来自单独一篇文档，不允许在两篇文档间拼接。
    - 只选择长度严格大于 context_size 的文档；不存在则 raise ValueError。
    - 随机选择 start 时，记住 torch.randint(low, high) 中 high 不包含。
    - x = doc[start:start+T], y = doc[start+1:start+T+1]。
    - 若 batch_size/context_size 非正则 raise ValueError。
    - 返回 dtype=torch.long 的两个张量。
    """
    if batch_size <= 0 or context_size <= 0:
        raise ValueError("batch_size and context_size must be positive")
    x_sample = []
    y_sample = []
    valid_docs = [doc for doc in documents
                if len(doc)>context_size
               ]
    if not valid_docs:
        raise ValueError("No valid documents")
    for _ in range(batch_size):
        random_int = torch.randint(low=0, high=len(valid_docs),size=(1,)).item()
        doc = valid_docs[random_int]
        start = torch.randint(0, len(doc) - context_size, (1,),dtype=torch.long).item()
        x_sample.append(doc[start:start+context_size])
        y_sample.append(doc[start+1:start+context_size+1])
    return torch.stack(x_sample, dim=0), torch.stack(y_sample, dim=0)




@torch.no_grad()
def evaluate_nll(model, documents: list[torch.Tensor], context_size: int) -> float:
    """TODO 2：在固定验证文档上，计算 deterministic、token-weighted NLL。

    推荐实现：
    1. 记录 was_training = model.training，然后 model.eval()。
    2. 对每篇 doc，按照 start=0,T,2T,... 切分 doc[:-1] 与 doc[1:]。
       x: doc[start:end], y: doc[start+1:end+1]，end=min(start+T,len(doc)-1)。
       每个 x.unsqueeze(0) shape (1, chunk_len)；模型返回 logits, _。
    3. 用 F.cross_entropy(..., reduction='sum') 得到每块 loss 总和。
    4. 全部文档的 total_loss / total_tokens（不是平均每块 loss）。
    5. 恢复 model 原先 train/eval 模式；不要更新参数。
    6. 若没有可评估的目标 token，应 raise ValueError。
    """
    # model.training 这是一个布尔属性（bool）：`True` / `False`
    # model.train() 和 model.eval() 本质就是修改这个变量！
    if context_size < 0:
        raise ValueError(f"context_size <UNK> context_size < 0")
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_sum = 0
    for doc in documents:
        start = 0
        end = min(len(doc), context_size)
        while start < len(doc) -1 :
            end = min(start + context_size, len(doc) - 1)
            x = doc[start:end]
            y = doc[start+1:end+1]
            start = start + context_size
            end = end + context_size
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            token_sum += y.numel() # len返回张量的第一维长度，numel返回张量元素总数
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)), y.reshape(-1),reduction="sum")
            loss_sum += loss
    loss_avg = loss_sum / token_sum
    model.train(was_training)
    return float(loss_avg)








def train_one_step(model, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """TODO 3：完成一次标准的 MiniGPT 参数更新。

    顺序：model.train() -> optimizer.zero_grad(set_to_none=True) ->
    _, loss = model(x, y) -> loss.backward() -> optimizer.step()。
    返回 float(loss.item())，检查 loss 是有限值。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    loss_val = loss.item()
    if not torch.isfinite(loss):
        raise ValueError("Loss is infinite or NaN!")
    return float(loss_val)


def update_early_stopping(best: float, current: float, stale: int,
                          *, patience: int = 4, min_delta: float = 1e-3):
    """TODO 4：实现 Early Stopping 状态机。

    improvement 条件：current < best - min_delta。
    若改善：更新 best，stale 置 0；否则 stale += 1。
    当 stale >= patience，should_stop=True。
    返回 (new_best, new_stale, should_stop, improved)。
    """
    if current < best - min_delta:
        best = current
        stale = 0
        improved = 1
    else:
        stale += 1
        improved = 0
    if stale >= patience:
        should_stop = True
    else:
        should_stop = False
    return best, stale, should_stop, improved


def preserve_best_weights(model, *, improved: bool, previous_state):
    """TODO 5：若验证 NLL 创新低，拷贝一份模型权重，用于结束时恢复。

    要求：
    - improved=False 时原样返回 previous_state。
    - improved=True 时对每个 state_dict 张量调用 detach().clone()。
    - 不要只存 model.state_dict() 引用（权重后续会继续改变）。
    - 返回的对象可以直接传给 model.load_state_dict(...)。
    """
    best_states = {}
    if not improved:
        return previous_state
    else:
        for name, weights in model.state_dict().items():
            best_states[name] = weights.detach().clone()
        return best_states


# ======================== 正确性测试（不要修改） ========================


def run_tests(MiniGPT):
    print("\n[TEST] 检查 TODO 1-5 ...")
    random.seed(23)
    torch.manual_seed(23)
    docs = [torch.arange(6, dtype=torch.long), torch.arange(10, 16, dtype=torch.long)]
    x, y = sample_batch(docs, 8, 5)
    assert x.shape == y.shape == (8, 5)
    assert x.dtype == torch.long and y.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1]), "错误：label 没有相对 input 偏移一位"
    assert all(int(row[0]) in {0, 10} for row in x), "错误：样本发生跨文档拼接"
    for row_x, row_y in zip(x, y):      # zip把多个可迭代对象打包，按照位置配对，返回迭代器
        assert torch.equal(row_y, row_x + 1)
    try:
        sample_batch([torch.arange(5)], 2, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("短文档应该 raise ValueError")

    class UniformModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
        def forward(self, ids):
            b, t = ids.shape
            return torch.zeros((b, t, self.vocab_size)), None

    uniform = UniformModel(17)
    loss_u = evaluate_nll(uniform, [torch.arange(11)], context_size=4)
    assert math.isclose(loss_u, math.log(17), rel_tol=1e-6), "NLL 应该按 token 数计算"

    model = MiniGPT(vocab_size=20, context_size=5, d_model=16,
                    num_heads=4, num_layers=1, dropout=0.2)
    model.train()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    nll_1 = evaluate_nll(model, [torch.arange(10)], context_size=5)
    nll_2 = evaluate_nll(model, [torch.arange(10)], context_size=5)
    assert math.isfinite(nll_1) and nll_1 == nll_2, "验证必须可复现且不包含 Dropout"
    assert model.training, "验证后应恢复之前的 model.train 模式"
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "验证修改了权重"

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train_loss = train_one_step(model, optimizer, x, y)
    assert math.isfinite(train_loss)
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items()), "没有更新参数"

    best, stale, stop, improved = update_early_stopping(float("inf"), 2.0, 0,
                                                        patience=2)
    assert (best, stale, stop, improved) == (2.0, 0, False, True)
    saved = preserve_best_weights(model, improved=True, previous_state=None)
    assert saved is not None
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(0.5)
    assert all(torch.equal(saved[k], expected[k]) for k in expected), "最佳权重必须深拷贝"
    assert preserve_best_weights(model, improved=False, previous_state=saved) is saved
    best, stale, stop, improved = update_early_stopping(best, 2.05, stale, patience=2)
    assert (best, stale, stop, improved) == (2.0, 1, False, False)
    best, stale, stop, improved = update_early_stopping(best, 2.1, stale, patience=2)
    assert (best, stale, stop, improved) == (2.0, 2, True, False)
    model.load_state_dict(saved)
    assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())
    print("[PASS] 5 个 TODO：采样、验证、训练、早停和权重恢复")


@dataclass(frozen=True)
class Experiment:
    name: str
    dropout: float
    weight_decay: float
    early_stop: bool


def run_experiment(MiniGPT, data: dict, spec: Experiment,
                   *, steps: int, eval_every: int, context_size: int,
                   batch_size: int, lr: float):
    # 每组实验使用相同初始化和文档划分；Dropout 打开后 RNG 消耗不同，是预期的。
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = MiniGPT(vocab_size=len(VOCAB), context_size=context_size,
                    d_model=32, num_heads=4, num_layers=2, dropout=spec.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=spec.weight_decay)
    logs, best, stale, saved, best_step = [], float("inf"), 0, None, None
    for step in range(1, steps + 1):
        x, y = sample_batch(data["train"], batch_size, context_size)
        train_one_step(model, opt, x, y)
        if step != 1 and step % eval_every and step != steps:
            continue
        train_nll = evaluate_nll(model, data["train"], context_size)
        val_nll = evaluate_nll(model, data["val"], context_size)
        logs.append({"experiment": spec.name, "step": step,
                     "train_nll": train_nll, "val_nll": val_nll,
                     "generalization_gap": val_nll - train_nll})
        print(f"{spec.name:>22} | step={step:>3} | "
              f"train={train_nll:.3f} | val={val_nll:.3f} | "
              f"gap={val_nll-train_nll:+.3f}")
        best, stale, stop, improved = update_early_stopping(
            best, val_nll, stale, patience=4, min_delta=1e-3
        )
        saved = preserve_best_weights(model, improved=improved, previous_state=saved)
        if improved:
            best_step = step
        if spec.early_stop and stop:
            break
    if saved is None:
        raise RuntimeError("尚未保存最佳权重")
    model.load_state_dict(saved)
    restored_val = evaluate_nll(model, data["val"], context_size)
    print(f"  Best validation step={best_step}, restored NLL={restored_val:.3f}")
    return logs


def save_results(rows: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "week02_metrics.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "step", "train_nll",
                                                 "val_nll", "generalization_gap"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装：已生成 CSV，跳过绘图")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sorted({row["experiment"] for row in rows}):
        subset = [r for r in rows if r["experiment"] == name]
        ax.plot([r["step"] for r in subset], [r["train_nll"] for r in subset],
                ls="--", label=f"{name}: train")
        ax.plot([r["step"] for r in subset], [r["val_nll"] for r in subset],
                label=f"{name}: val")
    ax.set(xlabel="Training step", ylabel="Character-level NLL",
           title="MiniGPT: training vs validation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "week02_losses.png", dpi=150)
    plt.close(fig)
    print(f"PNG: {OUTPUT_DIR / 'week02_losses.png'}")


def main():
    parser = argparse.ArgumentParser(description="Phase A / Week 02 student exercise")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or args.context_size < 1:
        parser.error("steps / eval-every / batch-size / context-size 必须大于 0")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    MiniGPT = load_module(MODEL_PATH, "student_minigpt_week02").MiniGPT
    run_tests(MiniGPT)
    if args.test_only:
        return
    data = prepare_documents()
    specs = [
        Experiment("baseline", 0.0, 0.0, False),
        Experiment("dropout_decay", 0.2, 0.1, False),
        Experiment("dropout_decay_ES", 0.2, 0.1, True),
    ]
    rows = []
    for spec in specs:
        rows += run_experiment(MiniGPT, data, spec,
                               steps=args.steps, eval_every=args.eval_every,
                               context_size=args.context_size,
                               batch_size=args.batch_size, lr=args.lr)
    save_results(rows)
    print("\n作业：解释每组曲线、最佳训练步数以及 20 篇短文档的统计局限。")


if __name__ == "__main__":
    main()
