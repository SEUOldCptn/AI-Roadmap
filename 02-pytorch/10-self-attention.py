import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]

        Q = self.W_q(x)  # [B, T, D]
        K = self.W_k(x)  # [B, T, D]
        V = self.W_v(x)  # [B, T, D]

        # [B, T, D] @ [B, D, T]
        # -> [B, T, T]
        scores = Q @ K.transpose(-2, -1)

        # Scale
        scores = scores / math.sqrt(Q.size(-1))

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # [B, T, T] @ [B, T, D]
        # -> [B, T, D]
        output = attention_weights @ V

        return output, attention_weights
x = torch.randn(2, 4, 8)

attention = SelfAttention(8)

y, attention_weights = attention(x)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("attention shape:", attention_weights.shape)
row_sums = attention_weights.sum(dim=-1)

print("row sums:")
print(row_sums)

# 验证self-attention的输出依赖整个输入序列
x = torch.randn(1, 4, 8)

y1, weights1 = attention(x)

x2 = x.clone()

x2[:, 2, :] += 5.0

y2, weights2 = attention(x2)

print("output difference:")
print((y1 - y2).abs().mean())

print("attention difference:")
print((weights1 - weights2).abs().mean())