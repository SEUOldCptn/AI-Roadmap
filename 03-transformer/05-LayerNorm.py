import torch
import torch.nn as nn

x = torch.tensor([
    [[1., 2., 3., 4.],
     [10., 20., 30., 40.]]
])

layer_norm = nn.LayerNorm(4)
# 创建一个层归一化（Layer Normalization）层，对最后一维长度为 4 的特征做归一化
# 里面的参数必须跟输入tensor的最后一维对齐

y = layer_norm(x)
# 对每一列分别归一化，不是对整个batch归一化

print("Output:", y)
print("Mean:", y.mean(dim=-1))
print("Variance:", y.var(dim=-1, unbiased=False))