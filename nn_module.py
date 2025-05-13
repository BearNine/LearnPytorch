import torch
from torch import nn


class Xiongxiong(nn.Module):
    def __init__(self):
        super(Xiongxiong, self).__init__()

    def forward(self, x):
        y = x + 1
        return y

xiongxiong = Xiongxiong()
xiong_x = torch.tensor(1)
out = xiongxiong(xiong_x)
print(out)