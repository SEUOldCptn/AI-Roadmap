import torch
import torch.nn as nn


# =====================================
# 1. 设置随机种子
# =====================================

torch.manual_seed(42)


# =====================================
# 2. XOR 数据
# =====================================

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


# =====================================
# 3. 定义神经网络
# =====================================

model = nn.Sequential(

    nn.Linear(2, 4),

    nn.Tanh(),

    nn.Linear(4, 1)
)


# =====================================
# 4. 损失函数
# =====================================

criterion = nn.BCEWithLogitsLoss()


# =====================================
# 5. 优化器
# =====================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1.0
)


# =====================================
# 6. 训练
# =====================================

epochs = 10000

for epoch in range(epochs):

    # -------------------------
    # Forward
    # -------------------------

    logits = model(X)

    loss = criterion(
        logits,
        y
    )


    # -------------------------
    # 清空旧梯度
    # -------------------------

    optimizer.zero_grad()       # PyTorch 默认梯度会累加，要先清理上一次计算的梯度


    # -------------------------
    # Backward
    # -------------------------
    print("Before:")
    print(model[0].weight)
    loss.backward()


    # -------------------------
    # 更新参数
    # -------------------------

    optimizer.step()


    print("After:")
    print(model[0].weight)

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:5d}, "
            f"Loss = {loss.item():.6f}"
        )


# =====================================
# 7. 推理
# =====================================

with torch.no_grad():

    logits = model(X)   # NN 的输出

    probabilities = torch.sigmoid(logits)   # 输出经过sigmoid之后的结果（概率）

    predictions = (
        probabilities >= 0.5
    ).int()


print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nTrue labels:")
print(y.int())


accuracy = (
    predictions == y
).float().mean()

print(
    f"\nAccuracy: "
    f"{accuracy.item() * 100:.2f}%"
)