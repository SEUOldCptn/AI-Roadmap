import time
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 模块 1：实现预分配静态 KV Cache 与动态 KV Cache 容器
# =====================================================================
class StaticKVCache:
    """
    预分配静态显存的 KV Cache 管理器（零 torch.cat 开销）
    """

    def __init__(
        self,
        max_batch_size: int,
        n_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype=torch.float32,
        device="cpu",
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # TODO 1.1: 使用 torch.zeros 一次性预分配 cache_k 和 cache_v
        # 形状要求: (max_batch_size, n_heads, max_seq_len, head_dim)
        self.cache_k = torch.zeros(max_batch_size,n_heads, max_seq_len, head_dim, dtype=dtype,device=device)
        self.cache_v = torch.zeros(max_batch_size,n_heads, max_seq_len, head_dim, dtype=dtype,device=device)

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将当前步新算出的 k_val, v_val 原位写入缓存，并返回截止当前位置 [0 : start_pos + T] 的有效 K, V 视图
        参数:
            k_val, v_val: shape 为 (B, H, T, D)
            start_pos: 当前输入序列在全局上下文中的起始位置索引 (int)
        返回:
            keys, values: shape 为 (B, H, start_pos + T, D)
        """
        B, H, T, D = k_val.shape
        assert (
            start_pos + T <= self.max_seq_len
        ), f"序列长度越界: {start_pos + T} > {self.max_seq_len}"

        # TODO 1.2: 使用张量切片将 k_val 和 v_val 原位写入 self.cache_k 和 self.cache_v
        # 提示: 注意处理当前实际批次大小 B <= max_batch_size 的情况
        ...
        self.cache_k[:,:,start_pos:start_pos+T,:] = k_val
        self.cache_v[:,:,start_pos:start_pos+T,:] = v_val

        # TODO 1.3: 切片返回从 0 到 start_pos + T 的有效历史 K 和 V 视图
        keys = self.cache_k[:,:,:start_pos+T,:]
        values = self.cache_v[:,:,:start_pos+T,:]
        return keys, values

    def memory_bytes(self) -> int:
        """返回当前 StaticKVCache 占用的实际字节数"""
        # TODO 1.4: 利用 numel() 和 element_size() 计算 cache_k + cache_v 的总字节数

        return 2*self.cache_k.numel()*self.cache_k.element_size()


class DynamicKVCache:
    """
    传统的基于 torch.cat 的动态拼接 KV Cache（用于对比性能与验证等价性）
    """

    def __init__(self):
        self.cache_k = None
        self.cache_v = None

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_pos == 0 or self.cache_k is None:
            self.cache_k = k_val
            self.cache_v = v_val
        else:
            # TODO 1.5: 使用 torch.cat 沿着序列长度维度 (dim=2) 拼接历史缓存与当前 k_val, v_val
            self.cache_k = torch.cat([self.cache_k, k_val], dim=-2)
            self.cache_v = torch.cat([self.cache_v, v_val], dim=-2)
        return self.cache_k, self.cache_v

class CausalSelfAttentionWithCache(nn.Module):

    def __init__(self, d_model: int, n_heads: int, context_size: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 注册下三角因果掩码: shape (context_size, context_size)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size)).view(
                1, 1, context_size, context_size
            ),
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache=None):
        B, T, C = x.shape

        # 1. 计算当前输入 x 的 q, k, v 并 reshape 为 (B, H, T, D)
        q = (
            self.q_proj(x)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 2. 如果启用了 kv_cache，更新并获取完整的历史 K, V
        if kv_cache is not None:
            # TODO 2.1: 调用 kv_cache.update 获取完整的 k, v (shape: B, H, start_pos + T, D)
            k, v = kv_cache.update(k,v,start_pos)
        else:
            assert start_pos == 0, "未启用 kv_cache 时，start_pos 必须为 0"

        total_seq_len = k.size(2)  # 等于 start_pos + T

        # 3. 计算注意力得分 scores: (B, H, T, total_seq_len)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # TODO 2.2: 使用我们在原理部分讲到的【通用切片公式】截取 self.causal_mask
        # 提示: 行范围是 [start_pos : start_pos + T]，列范围是 [: total_seq_len]
        mask = self.causal_mask[:, :, start_pos:start_pos+T, :total_seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ v  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# =====================================================================
# 模块 3：数值等价性验证（Prefill + Decode 逐词比对）与 GPU/CPU 计时器
# =====================================================================
@torch.no_grad()
def verify_kv_cache_equivalence(model, prompt_ids, gen_len, device, dtype):
    """
    核心验收测试：
    对比以下三种路径在生成过程中的每一步输出 Logits 是否严格一致：
      路径 A (Baseline): 每次将完整序列从头送入模型 (No Cache)
      路径 B (Dynamic):  Prefill + 逐 token Decode (使用 DynamicKVCache)
      路径 C (Static):   Prefill + 逐 token Decode (使用 StaticKVCache)
    """
    model.eval()
    B, T_prompt = prompt_ids.shape

    # TODO 3.1: 初始化路径 B 的 DynamicKVCache 列表（每层一个）和路径 C 的 StaticKVCache 列表
    max_seq_len = gen_len + T_prompt
    dynamic_caches = [DynamicKVCache() for _ in model.blocks]
    static_caches = [
        StaticKVCache(
            max_batch_size=B,
            n_heads=block.attn.n_heads,
            max_seq_len=max_seq_len,
            head_dim=block.attn.head_dim,
            dtype=dtype,
            device=device,
        ) for block in model.blocks
    ]
    # 1. 先执行路径 C (Static Cache) 的 Prefill 阶段 (start_pos = 0)
    # 拿到最后一个位置的 logits: [B, vocab_size]
    logits_static_step = model(
        prompt_ids, start_pos=0, kv_caches=static_caches
    )[:, -1, :]

    # 2. 执行路径 B (Dynamic Cache) 的 Prefill 阶段
    logits_dynamic_step = model(
        prompt_ids, start_pos=0, kv_caches=dynamic_caches
    )[:, -1, :]

    # 3. 逐步 Decode 循环，每一步都与全序列重算 (No Cache) 进行 torch.allclose 对账！
    full_seq = prompt_ids.clone()
    for step in range(gen_len):
        # 路径 A (No Cache): 直接把完整 full_seq 喂给模型，取最后一个位置的 logits
        logits_no_cache = model(full_seq, start_pos=0, kv_caches=None)[:, -1, :]

        # TODO 3.2: 使用 torch.allclose 验证 logits_no_cache 与 logits_static_step、logits_dynamic_step 是否一致
        # 注意: float32 下设 atol=1e-5，bfloat16 下设 atol=1e-2
        print("No_cache vs static_cache:",torch.allclose(logits_no_cache, logits_static_step, atol=1e-5))
        print("No_cache vs dynamic_cache:",torch.allclose(logits_no_cache, logits_dynamic_step, atol=1e-5))

        # 使用贪心采样 (argmax) 选出下一个 token: shape (B, 1)
        next_token = torch.argmax(logits_static_step, dim=-1, keepdim=True)
        cur_pos = full_seq.size(1)  # 当前新 token 的全局位置索引 start_pos
        full_seq = torch.cat([full_seq, next_token], dim=1)

        # TODO 3.3: 将 next_token (shape: B, 1) 和 start_pos=cur_pos 送入路径 B 和路径 C 进行单步 Decode
        if step < gen_len - 1:
            logits_static_step = model(next_token, start_pos=cur_pos, kv_caches=static_caches)[:, -1, :]
            logits_dynamic_step = model(next_token, start_pos=cur_pos, kv_caches=dynamic_caches)[:, -1, :]


def benchmark_generation(model, prompt_ids, gen_len, mode="static", warmup=3):
    """
    自适应 CPU / CUDA 的精确计时器
    mode: "no_cache" | "dynamic" | "static"
    """
    device = prompt_ids.device
    # 1. 预热阶段 (Warmup)
    for _ in range(warmup):
        generate_once(model, prompt_ids, gen_len, mode=mode)

    # 2. 正式计时阶段
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        # TODO 3.4: 执行指定 mode 的自回归生成
        generate_once(model, prompt_ids, gen_len, mode=mode)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        # 执行指定 mode 的自回归生成
        generate_once(model, prompt_ids, gen_len, mode=mode)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return elapsed_ms

@torch.no_grad()  # 重点：测速和推理时务必加上 no_grad，避免构建反向传播计算图！
def generate_once(model, prompt_ids: torch.Tensor, gen_len: int, mode: str):
    """
    根据指定的 mode ("no_cache" | "dynamic" | "static") 完整执行一次自回归生成
    """
    model.eval()
    B, T_prompt = prompt_ids.shape
    device = prompt_ids.device
    dtype = next(model.parameters()).dtype  # 获取模型当前的精度

    # ---------------- 模式 1：无缓存 (每次全序列从头算) ----------------
    if mode == "no_cache":
        full_seq = prompt_ids.clone()
        for _ in range(gen_len):
            logits = model(full_seq, start_pos=0, kv_caches=None)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            full_seq = torch.cat([full_seq, next_token], dim=1)
        return full_seq

    # ---------------- 模式 2 & 3：带缓存 (Prefill + 逐词 Decode) ----------------
    elif mode in ("dynamic", "static"):
        # 1. 初始化对应模式的 kv_caches 列表
        if mode == "dynamic":
            kv_caches = [DynamicKVCache() for _ in model.blocks]
        else:
            kv_caches = [
                StaticKVCache(
                    max_batch_size=B,
                    n_heads=block.attn.n_heads,
                    max_seq_len=T_prompt + gen_len,
                    head_dim=block.attn.head_dim,
                    dtype=dtype,
                    device=device,
                )
                for block in model.blocks
            ]

        # 2. Prefill 阶段：一次性送入整个 prompt_ids (start_pos = 0)
        logits = model(prompt_ids, start_pos=0, kv_caches=kv_caches)[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # 3. Decode 阶段：每次只送入刚生成的 1 个 next_token！
        cur_pos = T_prompt
        for _ in range(gen_len - 1):
            logits = model(next_token, start_pos=cur_pos, kv_caches=kv_caches)[
                :, -1, :
            ]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            cur_pos += 1

        return next_token
    else:
        raise ValueError(f"未知的 mode: {mode}")

# =====================================================================
# 模块 4：配套的 TransformerBlock 与 MiniGPT 模型定义（用于组装测试）
# =====================================================================
class TransformerBlockWithCache(nn.Module):

    def __init__(self, d_model: int, n_heads: int, context_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionWithCache(d_model, n_heads, context_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache=None):
        x = x + self.attn(self.ln1(x), start_pos=start_pos, kv_cache=kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPTWithCache(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        context_size: int,
    ):
        super().__init__()
        self.context_size = context_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlockWithCache(d_model, n_heads, context_size)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, start_pos: int = 0, kv_caches: list = None
    ):
        B, T = idx.shape
        assert (
            start_pos + T <= self.context_size
        ), f"超出模型最大上下文: {start_pos + T} > {self.context_size}"
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, start_pos=start_pos, kv_cache=cache)

        x = self.ln_f(x)
        return self.lm_head(x)


# =====================================================================
# 模块 5：Pytest 自动化单元测试 & 命令行运行入口
# =====================================================================
def test_kv_cache_shapes_and_memory():
    """测试 1：验证静态缓存的形状切片、越界保护与显存计算公式"""
    B, H, S_max, D = 2, 4, 16, 32
    cache = StaticKVCache(B, H, S_max, D, dtype=torch.float32, device="cpu")

    # 1. 验证显存字节数公式: 2 * B * H * S_max * D * 4 bytes
    expected_bytes = 2 * B * H * S_max * D * 4
    assert cache.memory_bytes() == expected_bytes

    # 2. 验证 Prefill (写入前 5 个 token) 后的返回形状
    k_prefill = torch.randn(B, H, 5, D)
    v_prefill = torch.randn(B, H, 5, D)
    k_out, v_out = cache.update(k_prefill, v_prefill, start_pos=0)
    assert k_out.shape == (B, H, 5, D)

    # 3. 验证单步 Decode (写入第 6 个 token) 后的返回形状
    k_decode = torch.randn(B, H, 1, D)
    v_decode = torch.randn(B, H, 1, D)
    k_out, v_out = cache.update(k_decode, v_decode, start_pos=5)
    assert k_out.shape == (B, H, 6, D)

    # 4. 验证超出 max_seq_len 时是否正确触发 AssertionError
    with pytest.raises(AssertionError):
        cache.update(k_prefill, v_prefill, start_pos=15)  # 15 + 5 = 20 > 16


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_equivalence_all_paths(device):
    """测试 2：验证 No-Cache、Dynamic-Cache、Static-Cache 三条路径数值严格等价"""
    torch.manual_seed(42)
    model = MiniGPTWithCache(
        vocab_size=200, d_model=64, n_heads=4, n_layers=2, context_size=128
    ).to(device)
    prompt_ids = torch.randint(0, 200, (2, 10), device=device)
    verify_kv_cache_equivalence(
        model, prompt_ids, gen_len=20, device=device, dtype=torch.float32
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f"🚀 当前运行设备: {device.upper()}")
    if device == "cuda":
        print(f"💳 GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"============================================================\n")

    torch.manual_seed(42)
    vocab_size = 1000
    d_model = 256
    n_heads = 8
    n_layers = 6
    context_size = 512

    model = MiniGPTWithCache(
        vocab_size, d_model, n_heads, n_layers, context_size
    ).to(device)
    prompt_ids = torch.randint(0, vocab_size, (4, 32), device=device)

    print("🔍 [Step 1] 正在执行 16 步生成数值等价性对账...")
    verify_kv_cache_equivalence(
        model, prompt_ids, gen_len=16, device=device, dtype=torch.float32
    )

    print("\n⏱️  [Step 2] 正在执行生成耗时对比压测 (Prompt=32, Gen=128, Batch=4)...")
    gen_len = 128
    t_no_cache = benchmark_generation(
        model, prompt_ids, gen_len, mode="no_cache", warmup=2
    )
    t_dynamic = benchmark_generation(
        model, prompt_ids, gen_len, mode="dynamic", warmup=2
    )
    t_static = benchmark_generation(
        model, prompt_ids, gen_len, mode="static", warmup=2
    )

    print(f"\n📊 压测结果汇总 (生成 {gen_len} tokens):")
    print(f"  - 路径 A (No Cache, O(N^3))       : {t_no_cache:8.2f} ms")
    print(
        f"  - 路径 B (Dynamic torch.cat)      : {t_dynamic:8.2f} ms  (加速比: {t_no_cache / t_dynamic:.2f}x)"
    )
    print(
        f"  - 路径 C (Static Pre-allocated)   : {t_static:8.2f} ms  (加速比: {t_no_cache / t_static:.2f}x)"
    )