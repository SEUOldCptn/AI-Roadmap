import tiktoken
import torch

input = torch.randn(10,20)
index = torch.randint(0,4, (4,), dtype=torch.long)
print(index)
data = torch.randn(10,20)
context_size = 5
batch_size = 2
index = torch.randint(0,len(data) - context_size, (batch_size,), dtype=torch.long)
list = torch.arange(context_size)
print(index[:,None]+list[None,:])
