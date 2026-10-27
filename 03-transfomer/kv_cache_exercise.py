"""AI-Roadmap 下一课：手写 KV Cache（学生 TODO 练习文件）.

运行：完成 TODO 后 python 09-kv-cache-exercise.py
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
        # TODO 1: 完成单层自注意力的 KV Cache
        # 输入 x: (B,T,D), past_kv=(past_k,past_v) 或 None
        # 输出: (attn_output, (all_k, all_v) if use_cache else None)
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split_heads(t):
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        # TODO 1a: 若已有缓存，将历史 k/v 与新 k/v 在 dim=-2 拼接
        if past_kv is not None:
            past_k, past_v = past_kv    # past_kv 是元组，直接拆分
            all_k = torch.cat([past_k, k], dim=-2)
            all_v = torch.cat([past_v, v], dim=-2)
        else:
            all_k = k
            all_v = v
        # 计算新增的token数和总token数
        T_new = q.size(-2)
        T_total = all_k.size(-2)
        past_len = T_total - T_new
        score = q @ all_k.transpose(-2, -1)/math.sqrt(self.head_dim)
        if T_new > 1:
            k_position = torch.arange(T_total,device=x.device)          # key 所有token的编号
            q_position = past_len + torch.arange(T_new,device=x.device)    # 新的token允许query最大的token编号
            mask = k_position[None, :] <= q_position[:, None]           # 长方形下三角矩阵
            score = score.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(score, dim=-1)
        output = weights @ all_v
        def combine_heads(x):
            x = x.transpose(1, 2)
            return x.reshape(B, T, -1)
        new_kv = (all_k, all_v) if use_cache else None
        return self.out_proj(combine_heads(output)), new_kv
        # TODO 1b: q @ all_k.transpose(-2,-1) / sqrt(head_dim)
        # TODO 1c: 建立绝对位置的因果掩码（包含 past_len 偏移）
        # TODO 1d: softmax、加权求和、合并 Heads、out_proj
        # TODO 1e: use_cache=True 时返回 all_k/all_v


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, past_kv=None, use_cache=False):
        # TODO 2: Pre-LN 残差连接；把 past_kv 交给 attention
        # 注意缓存只来自 attention，不缓存 FFN 结果
        # 要返回 (x, new_kv)
        x_res, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + x_res
        return x + self.ffn(self.ln2(x)), new_kv


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

    def forward(self, input_ids, target=None, past_kvs=None, use_cache=False):
        B,T = input_ids.shape
        if past_kvs is None:
            # TODO 3a: 每层有独立 (K,V) 缓存; 没有缓存时用 [None]*num_layers
            past_kvs = [None] * len(self.blocks)        # len(blocks) block中有多少层，就输出多少
        assert len(past_kvs) == len(self.blocks)
        past_len = (0 if past_kvs[0] is None else past_kvs[0][0].size(-2))
        # past_kvs is None：检查整个列表是否不存在。
        # past_kvs[0] is None：检查第0层是否尚未建立KVCache。
        if past_len + T > self.context_size:
            raise ValueError('past_len + T > context_size')
        # TODO 3b: position_ids 必须从 past_len 开始，而不是每次从 0 开始
        position_ids = past_len + torch.arange(T, device=input_ids.device)
        position_embeddings = self.position_embedding(position_ids)
        token_embeddings = self.token_embedding(input_ids)
        x = token_embeddings + position_embeddings

        new_past_kv = []
        for i, block in enumerate(self.blocks):
            x, new_kv = block(x, past_kv=past_kvs[i], use_cache=use_cache)
            new_past_kv.append(new_kv)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, -1),
                target.reshape(B * T)
            )
        if use_cache:
            # 带kvcache的推理
            return logits, new_past_kv
        else:
            # 训练/普通推理
            return logits, loss



        # TODO 3c: 把缓存逐层传递，并收集新的缓存
        # TODO 3d: 经 ln_f 和 lm_head 输出 logits (B,T,V)
        # 提示：若 past_len + T > context_size，本课直接抛 ValueError



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
    # TODO 4: Prefill 整段 prompt；Decode 阶段每次只输入新 token
    # 提示：先取得 logits, past_kvs = model(input_ids, use_cache=True)
    logits, past_kvs = model(ids,use_cache=True)
    # 然后 argmax(logits[:,-1,:])，下一次 model(next_id, past_kvs, use_cache=True)
    for _ in range(max_new_tokens):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)
        if _ == max_new_tokens-1:
            break
        logits, past_kvs = model(next_id, past_kvs=past_kvs, use_cache=True)
    # 返回形状 (B, prompt_len + max_new_tokens) 的 token IDs
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
