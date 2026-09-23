import torch


x = torch.tensor(2.0)

w = torch.tensor(
    3.0,
    requires_grad=True
)

b = torch.tensor(
    1.0,
    requires_grad=True
)


y = w * x + b

loss = y ** 2


print("y =", y)
print("loss =", loss)


loss.backward()


print("dL/dw =", w.grad)
print("dL/db =", b.grad)