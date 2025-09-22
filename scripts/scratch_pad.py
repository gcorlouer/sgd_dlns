import torch

x = torch.randn((3,3))
t = torch.triu(x, diagonal=1)
print(torch.diag(t))