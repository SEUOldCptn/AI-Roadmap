import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

set_seed(42)

def load_config(path):

    with open(path, "r", encoding="utf-8") as f:

        config = json.load(f)

    return config
config = load_config("config.json")
print(config)

# **全局基础配置，只需要在程序入口执行一次**，用来配置 root 根日志器：
#
# - `level=logging.INFO`：日志级别阈值。只输出 **INFO 及更高级别**（INFO, WARNING, ERROR, CRITICAL）；DEBUG 级别会直接过滤不打印。
# 级别顺序（由低到高）：`DEBUG < INFO < WARNING < ERROR < CRITICAL`
# - `format="%(asctime)s | %(levelname)s | %(message)s"`：日志输出模板
#   - `%(asctime)s`：打印时间戳
#   - `%(levelname)s`：日志等级名称（INFO/WARNING 等）
#   - `%(message)s`：你传入的日志内容
#
# 输出样例：
#
# ```
# 2026-10-02 15:30:22,123 | INFO | start training...

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def create_dataset(num_samples):

    class_0 = (
        torch.randn(num_samples, 2) * 0.8
        + torch.tensor([-2.0, 0.0])
    )

    class_1 = (
        torch.randn(num_samples, 2) * 0.8
        + torch.tensor([0.0, 2.0])
    )

    class_2 = (
        torch.randn(num_samples, 2) * 0.8
        + torch.tensor([2.0, 0.0])
    )

    X = torch.cat(
        [class_0, class_1, class_2],
        dim=0
    )

    y = torch.cat([
        torch.zeros(num_samples),
        torch.ones(num_samples),
        torch.full((num_samples,), 2)
    ]).long()

    return TensorDataset(X, y)

dataset = create_dataset(
    config["num_samples_per_class"]
)

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)   # 随机将数据按比例分为训练集和验证集

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False
)

class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_dim=2,
        hidden_dim=64,
        num_classes=3,
        dropout=0.3
    ):

        super().__init__()
        # `super().__init__()`
        # 的作用：
        # ✅ ** 调用父类
        # `nn.Module`
        # 的构造函数，完成
        # PyTorch
        # 底层的初始化工作 **。
        # 比如：注册网络参数、管理所有子层、开启
        # autograd
        # 梯度追踪、支持
        # `.cuda()
        # ` / `.state_dict()
        # `保存模型等。

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.bn1 = nn.BatchNorm1d(
            hidden_dim
        )
        # `nn.BatchNorm1d`： ** 一维批量归一化层（Batch
        # Normalization，BN） ** ，用于对
        # 2
        # D
        # 张量
        # `[batch_size, feature_dim]`
        # 做归一化，一般放在全连接层（Linear）之后、激活函数之前。
        # 训练时使用当前bacth的mean和variance
        # 训练过程中逐渐统计整个训练集的分布

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.bn2 = nn.BatchNorm1d(
            hidden_dim
        )

        self.fc3 = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.bn2(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)

        return x

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()       # 开启模型训练模式
    ## 会受影响的层
#
#     1. ** `nn.Dropout` **
#     - `model.train()`：启用，训练时随机把部分元素置
#     0
#     - `model.eval()`：关闭，全部神经元保留
#
#
# 2. ** `nn.BatchNorm1d / BatchNorm2d` **
# - `model.train()`：使用当前
# batch
# 的均值、方差；并且 ** 更新滑动平均的全局均值方差 **（保存下来给推理用）
# - `model.eval()`：不再计算
# batch
# 的统计量，直接读取训练阶段保存好的全局均值方差

    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * X.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return average_loss, accuracy

def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()
    # 推理的时候不会使用当前batch的mean和var，而是
    # 用之前训练得到的
    # eval本身不会影响计算图的建立，所以需要torch.no_grad

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # 推理时不建立计算图，关闭梯度

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            total_loss += (
                loss.item()
                * X.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return average_loss, accuracy

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_loss
):
    checkpoint = {
        "epoch": epoch,  # 当前训练到第几轮epoch，恢复时可以从下一轮继续训练
        "model_state_dict": model.state_dict(),  # ✅模型权重（所有Linear、BN层参数），核心！
        "optimizer_state_dict": optimizer.state_dict(),  # ✅优化器状态：学习率、动量、Adam的一阶/二阶矩等
        "val_loss": val_loss,  # 当前验证集loss，用来判断是不是最好模型，方便保留最优权重
    }

    torch.save(
        checkpoint,
        path
    )

def load_checkpoint(
    path,
    model,
    optimizer=None
):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint["epoch"]

    val_loss = checkpoint["val_loss"]

    return epoch, val_loss

def create_optimizer(
    name,
    model,
    lr,
    weight_decay
):

    if name == "SGD":

        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    elif name == "Adam":

        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "AdamW":

        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    else:

        raise ValueError(
            f"Unknown optimizer: {name}"
        )

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device
):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            # 保留在验证集上loss最低的模型
            best_val_loss = val_loss

            save_checkpoint(
                "best_model.pt",
                model,
                optimizer,
                epoch,
                val_loss
            )

    return history

def plot_history(history):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "loss_curve.png"
    )

    plt.show()

def plot_accuracy(history):

    epochs = range(
        1,
        len(history["train_acc"]) + 1
    )

    plt.figure()

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "accuracy_curve.png"
    )

    plt.show()

def main():

    config = load_config(
        "config.json"
    )

    set_seed(
        config["seed"]
    )

    device = torch.device("cpu")

    logger.info(
        f"Using device: {device}"
    )

    dataset = create_dataset(
        config["num_samples_per_class"]
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    model = MLPClassifier(
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        config["optimizer"],
        model,
        config["learning_rate"],
        config["weight_decay"]
    )

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        config["epochs"],
        device
    )

    plot_history(history)

    plot_accuracy(history)


if __name__ == "__main__":

    main()
##### 程序结构 ######
# config.json
#        │
#        ↓
# load_config()
#        │
#        ↓
# set_seed()
#        │
#        ↓
# Dataset
#        │
#        ├─────────────┐
#        ↓             ↓
# TrainLoader       ValLoader
#        │             │
#        ↓             │
#       MLP            │
#        │             │
#        ↓             │
# train_one_epoch()    │
#        │             │
#        └────→ evaluate()
#                     │
#                     ↓
#                Validation
#                     │
#                     ↓
#              save_checkpoint()
#                     │
#                     ↓
#              best_model.pt

new_model = MLPClassifier(
    hidden_dim=config["hidden_dim"],
    dropout=config["dropout"]
)

new_optimizer = create_optimizer(
    config["optimizer"],
    new_model,
    config["learning_rate"],
    config["weight_decay"]
)

epoch, val_loss = load_checkpoint(
    "best_model.pt",
    new_model,
    new_optimizer
)

print(
    "Loaded epoch:",
    epoch
)

print(
    "Loaded val loss:",
    val_loss
)