import torch


# =============================
# 1. 创建 Tensor
# =============================

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("x:")
print(x)

print("\nshape:")
print(x.shape)

print("\ndtype:")
print(x.dtype)


# =============================
# 2. 创建特殊 Tensor
# =============================

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_tensor = torch.randn(2, 3)

print("\nzeros:")
print(zeros)

print("\nones:")
print(ones)

print("\nrandom:")
print(random_tensor)
# =============================
# 3. Tensor 运算
# =============================

A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

B = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

print("\nA + B:")
print(A + B)

print("\nElement-wise multiplication:")
print(A * B)

print("\nMatrix multiplication:")
print(A @ B)

print(A.shape)