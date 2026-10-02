import torch

# # x = torch.arange(24)
# #
# # print(x)
# # print(x.shape)
# #
# # x = x.reshape(2,3,4)
# # print(x)
# # print(x.shape)
#
# x = torch.arange(24)
#
# y = x.view(2, 3, 4)
#
# print(y)
# print(y.shape)
#
# x = torch.arange(24).reshape(2, 3, 4)
# y = x.permute(1,0,2)
# print(y.shape)
# # z = y.view(6,4) view只修改形状不改变内存
# # permute 之后数据逻辑连续，但是物理内存不连续，直接view会报错
# z = y.reshape(6,4)
# # reshape 按需求拷贝数据，数据逻辑和物理内存都连续
# print(z.shape)
#
# x = torch.arange(24).reshape(2, 3, 4)
#
# print(x.shape)
# print(x.stride())


a = torch.randn(4)
b = torch.randn(4)

c = torch.einsum("i,i->", a, b)

print(c)