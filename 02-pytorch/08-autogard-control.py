import torch

x = torch.tensor(2.0)

print(x.requires_grad)


x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x * 3

z = y.detach()

out = y ** 2 + z ** 2

out.backward()

print("x =", x)
print("y =", y)
print("z =", z)
print("x.grad =", x.grad)