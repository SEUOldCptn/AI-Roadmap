import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
torch.manual_seed(123)
num_samples = 100
class_0 = torch.randn(num_samples, 2) * 0.5 + torch.tensor([-2.0, 0.0])
class_1 = torch.randn(num_samples, 2) * 0.5 + torch.tensor([0.0, 2.0])
class_2 = torch.randn(num_samples, 2) * 0.5 + torch.tensor([2.0, 0.0])

X = torch.cat([class_0, class_1, class_2], dim=0)
y = torch.cat([torch.zeros(num_samples), torch.ones(num_samples), torch.full((num_samples,), 2)]).long()

dataset = TensorDataset(X, y)
sample_x, sample_y = dataset[0]
# print(X.shape)
# print(y.shape)
dataloader = DataLoader(dataset,batch_size=32,shuffle=True)     # 每次32个样本训练，shuffle表示每个epoch重新打乱样本
for batch_X, batch_y in dataloader:

    print(batch_X.shape)
    print(batch_y.shape)

    break
# Epoch
# 整个训练集完整经过模型一次。
# Iteration / Step
# 处理一个 batch。