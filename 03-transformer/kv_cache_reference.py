"""AI-Roadmap 下一课：手写 KV Cache（CPU 可运行的独立参考实现）.

运行：python 09-kv-cache-reference.py
依赖：pip install torch
前置知识：MiniGPT, causal attention, position embedding, torch.cat.

说明：固定绝对位置编码下，为确保严格等价，实验只在不超过 context_size
的序列内比较；长文本滑动窗口需处理位置重编号和缓存重建。
"""
import math
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None, use_cache=False):
        """x (B,T,D), past_kv 是前面时刻已经计算过的 (K,V)."""
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # 适用于 prefill (T>1) 和 decode (T=1) 的通用因果掩码。
        q_positions = past_len + torch.arange(T, device=x.device)
        k_positions = torch.arange(past_len + T, device=x.device)
        allowed = k_positions[None, :] <= q_positions[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().reshape(B, T, D)
        output = self.out_proj(context)
        new_kv = (k, v) if use_cache else None
        return output, new_kv


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=101, context_size=128, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, 4 * d_model) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, past_kvs=None, use_cache=False):
        B, T = input_ids.shape
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        if len(past_kvs) != len(self.blocks):
            raise ValueError("每层 Transformer 必须对应一个 KV Cache")
        if not use_cache and any(c is not None for c in past_kvs):
            raise ValueError("只有 use_cache=True 才能传入已有缓存")

        past_len = 0 if past_kvs[0] is None else past_kvs[0][0].size(-2)
        if past_len + T > self.context_size:
            raise ValueError("实验仅支持不超过 context_size 的绝对位置范围")

        pos = torch.arange(past_len, past_len + T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)

        new_kvs = []
        for block, cache in zip(self.blocks, past_kvs):
            x, new_cache = block(x, cache, use_cache)
            if use_cache:
                new_kvs.append(new_cache)
        logits = self.lm_head(self.ln_f(x))
        return logits, (new_kvs if use_cache else None)


@torch.inference_mode()
def generate_uncached(model, input_ids, max_new_tokens):
    model.eval()
    ids = input_ids.clone()
    if ids.size(1) + max_new_tokens > model.context_size:
        raise ValueError("本实验避免跨过 context_size；长文本需重建窗口缓存")
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)
    return ids


@torch.inference_mode()
def generate_cached(model, input_ids, max_new_tokens):
    model.eval()
    ids = input_ids.clone()
    if ids.size(1) + max_new_tokens > model.context_size:
        raise ValueError("本实验避免跨过 context_size；长文本需重建窗口缓存")
    # Prefill: 首次对整个 prompt 计算所有层的 K/V。
    logits, past_kvs = model(ids, use_cache=True)
    for step in range(max_new_tokens):
        # 当前最后一个 token 的预测分布对应下一个 token。
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)
        # 下次只把刚生成的 1 个 token 输入模型。
        if step < max_new_tokens - 1:
            logits, past_kvs = model(next_id, past_kvs=past_kvs, use_cache=True)
    return ids


@torch.inference_mode()
def test_equivalence():
    torch.manual_seed(42)
    model = MiniGPT(vocab_size=101, context_size=128).eval()
    prompt = torch.randint(0, 101, (2, 24))

    full_logits, _ = model(prompt)
    prefill_logits, kvs = model(prompt, use_cache=True)
    assert torch.allclose(full_logits, prefill_logits, atol=1e-5, rtol=1e-5)
    assert len(kvs) == len(model.blocks)
    assert kvs[0][0].shape == (2, 4, 24, 16)
    assert kvs[0][1].shape == (2, 4, 24, 16)

    next_id = torch.randint(0, 101, (2, 1))
    no_cache_logits, _ = model(torch.cat((prompt, next_id), dim=1))
    cache_logits, kvs = model(next_id, past_kvs=kvs, use_cache=True)
    error = (no_cache_logits[:, -1] - cache_logits[:, -1]).abs().max().item()
    assert torch.allclose(no_cache_logits[:, -1], cache_logits[:, -1], atol=1e-5, rtol=1e-5)
    assert kvs[0][0].shape[-2] == 25

    plain = generate_uncached(model, prompt, max_new_tokens=32)
    cached = generate_cached(model, prompt, max_new_tokens=32)
    assert torch.equal(plain, cached), "缓存不应改变 greedy decoding 结果"
    print(f"[PASS] Prefill/Decode logits 等价，最大误差 {error:.3e}")
    print("[PASS] 两种生成方式得到完全相同的 token 序列")
    return model, prompt


def benchmark(model, prompt, new_tokens=32, repeats=5):
    # 对比相同模型、输入、输出 token 数；只对 Python/CPU 运行环境有代表性。
    for _ in range(2):
        generate_uncached(model, prompt, new_tokens)
        generate_cached(model, prompt, new_tokens)
    times = {"uncached": [], "cached": []}
    for _ in range(repeats):
        start = time.perf_counter()
        generate_uncached(model, prompt, new_tokens)
        times["uncached"].append(time.perf_counter() - start)
        start = time.perf_counter()
        generate_cached(model, prompt, new_tokens)
        times["cached"].append(time.perf_counter() - start)
    a = statistics.median(times["uncached"])
    b = statistics.median(times["cached"])
    print(f"No Cache: {a*1000:.1f} ms | KV Cache: {b*1000:.1f} ms | speedup: {a/b:.2f}x")
    print("注意：很小的 CPU 模型可能因缓存拼接和 Python 开销而不加速，不能直接代表 GPU LLM。")


if __name__ == "__main__":
    torch.set_num_threads(1)
    model, prompt = test_equivalence()
    benchmark(model, prompt)
