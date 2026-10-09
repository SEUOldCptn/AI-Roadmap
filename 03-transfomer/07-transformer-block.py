"""AI-Roadmap · Phase B: Transformer Decoder Block (CPU friendly).

Prerequisite: multi-head attention, causal mask, positional encoding.
Run: python 04-transformer-block.py
"""

import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a lower-triangular causal mask."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, L, D) -> (B, H, L, D/H)
            return t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # (L, L) broadcasts to (B, H, L, L)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v  # (B, H, L, head_dim)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        return self.out_proj(context)




class FeedForward(nn.Module):
    """Same token-wise MLP is applied independently at every position."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            # GELU(x)=x*PHI(x) PHI是标准正态分布的cdf
            # 相比RELU的好处：
            #     - 处处连续可导
            #     - 负数区域梯度 ** 不会直接变成0 **，只是很小
            #      - 不会出现硬死亡，训练稳定性更好，尤其深层网络（Transformer几十上百层）
            nn.Linear(d_ff, d_model),
        )
    """
    FFN和Attention的分工：
    Attention让token和其他token关联，找到其他token携带的信息
    FFN从该token与其他token关联的信息中提取特征(区分一词多义)
    FFN对每个输入都进行一样的变换
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    """Pre-LN GPT-style decoder block: LN->Attention->Add, LN->FFN->Add."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        # d_ff: FFN隐藏层维度
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.resid_dropout = nn.Dropout(dropout)
        # 残差分支dropout，防止模型过拟合

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + residual around causal attention
        x = x + self.resid_dropout(self.attn(self.ln1(x)))
        # Pre-LN + residual around position-wise FFN
        x = x + self.resid_dropout(self.ffn(self.ln2(x)))
        return x


def main():
    torch.manual_seed(42)
    batch_size, seq_len, d_model, num_heads = 2, 6, 32, 8
    model = TransformerDecoderBlock(d_model, num_heads, 4 * d_model, dropout=0.1)

    # 1) Forward and gradient tests
    model.train()
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert model.attn.qkv.weight.grad is not None
    assert model.ffn.net[0].weight.grad is not None
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    print("Attention gradient norm:", model.attn.qkv.weight.grad.norm().item())
    print("FFN gradient norm:", model.ffn.net[0].weight.grad.norm().item())

    # 2) Causal test: changing positions >=4 must not affect positions <4
    model.eval()  # important: disable dropout for a deterministic comparison
    with torch.no_grad():
        original = torch.randn(1, seq_len, d_model)
        changed = original.clone()
        changed[:, 4:, :] += torch.randn_like(changed[:, 4:, :]) * 10.0
        original_output = model(original)
        changed_output = model(changed)
        past_difference = (original_output[:, :4] - changed_output[:, :4]).abs().max().item()
        assert torch.allclose(original_output[:, :4], changed_output[:, :4], atol=1e-6)
        print("Max difference in earlier tokens after future-token change:", past_difference)

    # 3) Residual identity check when both sublayer outputs are exactly zero
    identity_model = TransformerDecoderBlock(d_model, num_heads, 4 * d_model, dropout=0.0)
    with torch.no_grad():
        for layer in (identity_model.attn, identity_model.ffn):
            for p in layer.parameters():
                p.zero_()
        identity_output = identity_model(original)
        assert torch.allclose(identity_output, original)
        print("Zero-sublayer residual identity test: PASS")

    print("All tests passed.")

    lynorm = nn.LayerNorm(d_model)
    x_norm = lynorm(x)
    print("mean:", x_norm.mean())
    print("var:", x_norm.var())


if __name__ == "__main__":
    main()
