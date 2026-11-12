"""AI-Roadmap Phase B: build and train a tiny autoregressive GPT on CPU.

Run: python 08-mini-gpt.py
Requirements: torch

The Transformer components are the same Pre-LN + causal attention design
studied in 07-transformer-block.py. Kept in this file to make it runnable alone.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.ln1(x)))
        x = x + self.resid_dropout(self.ffn(self.ln2(x)))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, context_size=16, d_model=32,
                 num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerDecoderBlock(d_model, num_heads, 4 * d_model, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, targets=None):
        B, T = input_ids.shape
        if T > self.context_size:
            raise ValueError("Input sequence exceeds context_size")
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(B * T, -1), targets.reshape(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens=30,
                 temperature=1.0, top_k=1):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.context_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                top_values, _ = torch.topk(next_logits, k=k)
                next_logits = next_logits.masked_fill(
                    next_logits < top_values[:, [-1]], float("-inf")
                )
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


def sample_batch(data, batch_size: int, context_size: int):
    # x = corpus[i : i+T], y = corpus[i+1 : i+T+1]
    starts = torch.randint(0, len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i + context_size] for i in starts])
    y = torch.stack([data[i + 1:i + context_size + 1] for i in starts])
    return x, y


def test_causality(model, vocab_size):
    model.eval()  # disables dropout
    original = torch.randint(0, vocab_size, (1, 8))
    modified = original.clone()
    modified[:, 4:] = (modified[:, 4:] + 1) % vocab_size
    with torch.no_grad():
        logits_a, _ = model(original)
        logits_b, _ = model(modified)
    diff = (logits_a[:, :4] - logits_b[:, :4]).abs().max().item()
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)
    print(f"Causality test: PASS (maximum prefix difference: {diff:.2e})")


def main():
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    # Character-level tokenization. This repeated toy dataset is for learning,
    # not for evaluating real-world generalization.
    corpus = "hello world!\n" * 200
    chars = sorted(set(corpus))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    encode = lambda s: [char_to_id[ch] for ch in s]
    decode = lambda ids: "".join(id_to_char[int(i)] for i in ids)
    data = torch.tensor(encode(corpus), dtype=torch.long)

    context_size, batch_size = 16, 32
    model = MiniGPT(vocab_size=len(chars), context_size=context_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print("Vocabulary:", repr("".join(chars)))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    x, y = sample_batch(data, batch_size, context_size)
    print("Input shape:", tuple(x.shape), "Target shape:", tuple(y.shape))
    print("One shifted example:", repr(decode(x[0])), "->", repr(decode(y[0])))
    print("Initial output:", repr(decode(model.generate(torch.tensor([encode("hello")]), 24)[0])))

    test_causality(model, vocab_size=len(chars))
    for step in range(1, 301):
        model.train()
        inputs, targets = sample_batch(data, batch_size, context_size)
        logits, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step {step:3d} | loss = {loss.item():.4f}")

    prompt = torch.tensor([encode("hello")], dtype=torch.long)
    output_ids = model.generate(prompt, max_new_tokens=40, top_k=1)
    print("Generated text:", repr(decode(output_ids[0])))
    test_causality(model, vocab_size=len(chars))


if __name__ == "__main__":
    main()
