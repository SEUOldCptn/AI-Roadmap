import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


# =====================================
# 1. Random seed
# =====================================

torch.manual_seed(42)


# =====================================
# 2. Create dataset
# =====================================

num_samples = 100


class_0 = (
    torch.randn(num_samples, 2) * 0.5
    +
    torch.tensor([-2.0, 0.0])
)


class_1 = (
    torch.randn(num_samples, 2) * 0.5
    +
    torch.tensor([0.0, 2.0])
)


class_2 = (
    torch.randn(num_samples, 2) * 0.5
    +
    torch.tensor([2.0, 0.0])
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


# =====================================
# 3. Dataset and DataLoader
# =====================================

dataset = TensorDataset(X, y)   # dataset是数据集的定义


dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
# dataloader负责每次迭代从dataset中取样

# =====================================
# 4. Define model
# =====================================

class SimpleClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(2, 16)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 8)

        self.fc3 = nn.Linear(8, 3)


    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.fc3(x)

        return x


model = SimpleClassifier()


# =====================================
# 5. Loss function
# =====================================

criterion = nn.CrossEntropyLoss()


# =====================================
# 6. Optimizer
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# =====================================
# 7. Training
# =====================================

epochs = 100


for epoch in range(epochs):

    total_loss = 0.0

    correct = 0

    total = 0


    for batch_X, batch_y in dataloader:

        # -----------------------------
        # Clear gradients
        # -----------------------------

        optimizer.zero_grad()


        # -----------------------------
        # Forward
        # -----------------------------

        logits = model(batch_X)


        # -----------------------------
        # Loss
        # -----------------------------

        loss = criterion(
            logits,
            batch_y
        )


        # -----------------------------
        # Backward
        # -----------------------------

        loss.backward()


        # -----------------------------
        # Update parameters
        # -----------------------------

        optimizer.step()


        # -----------------------------
        # Statistics
        # -----------------------------

        total_loss += (
            loss.item()
            *
            batch_X.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            predictions == batch_y
        ).sum().item()


        total += batch_y.size(0)


    average_loss = (
        total_loss / total
    )


    accuracy = (
        correct / total
    )


    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )


# =====================================
# 8. Evaluation
# =====================================

model.eval()


with torch.no_grad():

    logits = model(X)

    predictions = torch.argmax(
        logits,
        dim=1
    )

    accuracy = (
        predictions == y
    ).float().mean()


print(
    f"\nFinal accuracy: "
    f"{accuracy.item() * 100:.2f}%"
)
total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(total_params)