import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # 为了方便学习，这里要求 d_model 为偶数
        assert d_model % 2 == 0

        # (L, D)
        pe = torch.zeros(
            max_len,
            d_model
        )

        # position:
        # (L, 1)
        #
        # [[0],
        #  [1],
        #  [2],
        #  ...]
        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        # 对应公式中的
        #
        # 1 / 10000^(2i / d_model)
        #
        # shape: (D / 2,)
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            )
            * (-math.log(10000.0) / d_model)
        )
        # PE(pos,2i) = sin(pos/1000^(2i/dmodel)
        # PE(pos,2i+1) = cos(pos/1000^(2i/dmodel)
        # i = 0,1,2,...,dmodel/2-1
        # i：代表特征向量里 ** 偶数下标维度 **：\(0, 2, 4...\)
        # - \(2i+1\)：代表 ** 奇数下标维度 **：\(1, 3, 5...\)
        #
        # 举例，当 \(d_{model}=512\)：
        # i
        # 的取值范围：\(0, 1, 2, ..., 255\)，一共
        # 256
        # 个。
        # 偶数维使用 sin

        ##### 不同维度的频率不一样，不同pos则代表序列位置不一样(第几个token) #####
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # 奇数维使用 cos
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # 增加 batch 维度
        #
        # (L, D)
        #     ↓
        # (1, L, D)
        pe = pe.unsqueeze(0)
        # unqueeze(x) 在第x维增加一个维度
        # PE 不是可训练参数，
        # 但希望能够跟着 model.to(device) 移动
        self.register_buffer(
            "pe",
            pe
        )
        # pe也是模型的一部分，要跟模型放在一个内存中
        # 但是buffer不是parameter不需要训练，只需要跟随device移动

    def forward(self, x):

        # x:
        # (B, L, D)

        seq_len = x.size(1)

        return (
            x
            + self.pe[:, :seq_len, :]
        )


class TokenAndPositionEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        max_len
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_len,
            d_model
        )

    def forward(self, input_ids):

        # input_ids:
        # (B, L)

        B, L = input_ids.shape

        token_emb = self.token_embedding(
            input_ids
        )

        # [0, 1, 2, ..., L-1]
        positions = torch.arange(
            L,
            device=input_ids.device
        )

        pos_emb = self.position_embedding(
            positions
        )

        # token_emb:
        # (B, L, D)
        #
        # pos_emb:
        # (L, D)
        #
        # PyTorch 自动 broadcast
        return token_emb + pos_emb


def test_sinusoidal():

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 8

    x = torch.zeros(
        batch_size,
        seq_len,
        d_model
    )

    pe = SinusoidalPositionalEncoding(
        d_model=d_model,
        max_len=32
    )

    output = pe(x)

    print(
        "\nSinusoidal PE shape:",
        output.shape
    )

    print(
        "\nPosition 0:"
    )
    print(output[0, 0])

    print(
        "\nPosition 1:"
    )
    print(output[0, 1])

    print(
        "\nPosition 2:"
    )
    print(output[0, 2])


def test_learned_embedding():

    torch.manual_seed(42)

    vocab_size = 100
    d_model = 8
    max_len = 32

    model = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len
    )

    # 故意让所有 token 都一样
    input_ids = torch.tensor([
        [5, 5, 5, 5]
    ])

    token_only = model.token_embedding(
        input_ids
    )

    output = model(
        input_ids
    )

    print(
        "\nToken embedding:"
    )

    print(token_only)

    print(
        "\nToken 0 和 Token 1 原始 embedding 是否一样:"
    )

    print(
        torch.allclose(
            token_only[0, 0],
            token_only[0, 1]
        )
    )

    print(
        "\n加入 position embedding 后是否一样:"
    )

    print(
        torch.allclose(
            output[0, 0],
            output[0, 1]
        )
    )

    print(
        "\nFinal embedding shape:",
        output.shape
    )


if __name__ == "__main__":

    test_sinusoidal()

    test_learned_embedding()