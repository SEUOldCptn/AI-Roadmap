import numpy as np


# ==============================
# 1. 准备训练数据
# ==============================

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


# ==============================
# 2. 初始化神经网络参数
# ==============================

np.random.seed(42)

input_size = 2
hidden_size = 8
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))


# ==============================
# 3. 定义激活函数
# ==============================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ==============================
# 4. 设置训练参数
# ==============================

learning_rate = 1
epochs = 10000

epsilon = 1e-8


# ==============================
# 5. 开始训练
# ==============================

for epoch in range(epochs):

    # ---------- 前向传播 ----------

    # 第一层：输入层 -> 隐藏层
    Z1 = X @ W1 + b1

    # 隐藏层激活函数
    A1 = np.tanh(Z1)

    # 第二层：隐藏层 -> 输出层
    Z2 = A1 @ W2 + b2

    # 输出概率
    y_hat = sigmoid(Z2)


    # ---------- 计算损失 ----------

    loss = -np.mean(
        y * np.log(y_hat + epsilon)
        +
        (1 - y) * np.log(1 - y_hat + epsilon)
    )


    # ---------- 反向传播 ----------

    m = X.shape[0]

    # 输出层梯度
    dZ2 = (y_hat - y) / m

    dW2 = A1.T @ dZ2

    db2 = np.sum(
        dZ2,
        axis=0,
        keepdims=True
    )


    # 隐藏层梯度
    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * (1 - A1 ** 2)

    dW1 = X.T @ dZ1

    db1 = np.sum(
        dZ1,
        axis=0,
        keepdims=True
    )


    # ---------- 梯度下降 ----------

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2


    # 每1000轮打印一次
    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:5d}, "
            f"Loss = {loss:.6f}"
        )


# ==============================
# 6. 查看最终预测结果
# ==============================

print("\nTraining finished.")
print("\nPredicted probabilities:")

print(y_hat)


predictions = (y_hat >= 0.5).astype(int)

print("\nPredicted labels:")

print(predictions)


print("\nTrue labels:")

print(y.astype(int))

accuracy = np.mean(predictions == y)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)