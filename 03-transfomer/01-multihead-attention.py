
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, (
            "d_model 必须能够被 num_heads 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 对所有 Head 一次性执行线性投影。
        # 每个 Head 对应不同的投影参数切片。
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 合并 Head 后进行输出投影
        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x: (B, L, d_model)
        B, L, _ = x.shape

        x = x.reshape(
            B, L, self.num_heads, self.head_dim
        )

        # (B, L, H, D) -> (B, H, L, D)
        x = x.permute(0, 2, 1, 3)

        return x

    def merge_heads(self, x):
        # x: (B, H, L, D)
        B, H, L, D = x.shape

        # 恢复到 (B, L, H, D)
        x = x.permute(0, 2, 1, 3)

        # 先 contiguous，再 reshape
        x = x.contiguous().reshape(
            B, L, H * D
        )

        return x

    def forward(self, x):
        # x: (B, L, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 每个 Head 独立计算注意力分数
        # scores = torch.matmul(
        #     Q, K.transpose(-2, -1)
        # ) / math.sqrt(self.head_dim)
        scores = torch.einsum("bhtd,bhsd->bhts",
                              Q, K)/math.sqrt(self.head_dim)

        # 在 Key 所对应的序列维度上归一化
        attn_weights = torch.softmax(
            scores, dim=-1
        )

        # 根据注意力权重融合 Value
        context = torch.matmul(
            attn_weights, V
        )

        # 合并 Head
        context = self.merge_heads(context)

        # 输出投影
        output = self.out_proj(context)

        return output, attn_weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 4

    # 随机生成一批输入序列
    x = torch.randn(
        batch_size, seq_len, d_model
    )

    model = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    output, weights = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)

    # 检查每一行注意力权重之和是否为 1
    print(
        "Attention row sums:",
        weights[0, 0].sum(dim=-1)
    )

    # 简单测试反向传播
    loss = output.square().mean()
    loss.backward()

    print(
        "Q projection gradient norm:",
        model.q_proj.weight.grad.norm().item()
    )
    for head in range(num_heads):
        print(f"\nHead {head} attention:")
        print(weights[0, head].detach())


if __name__ == "__main__":
    main()
