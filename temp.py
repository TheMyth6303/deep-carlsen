import torch.nn as nn
import torch
loss = nn.MSELoss()
print(loss(torch.Tensor([1]), torch.Tensor([3])))
