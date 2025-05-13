import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_data = torchvision.datasets.CIFAR10(root='./cifdataset', train=False, transform=transforms.ToTensor(), download=True)
test_data = torch.utils.data.DataLoader(test_data, batch_size=1)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float)
input = torch.reshape(input, (-1, 1, 5, 5))

class Juan(nn.Module):
    def __init__(self):
        super(Juan, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (3,3), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Xiong_Pool(nn.Module):
    def __init__(self):
        super(Xiong_Pool, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), ceil_mode=False)

    def forward(self, x):
        x = self.pool1(x)
        return x

class Xiong_ReLU(nn.Module):
    def __init__(self):
        super(Xiong_ReLU, self).__init__()

    def forward(self, x):
        x = nn.ReLU(inplace=True)(x)
        return x

xiong_juan = Juan()


writer = SummaryWriter('logs')
for i, datajuan in enumerate(test_data):
    imgs, targets = datajuan
    out = xiong_juan(imgs)
    print(out.shape)
    out = torch.reshape(out, (-1, 3, 30, 30))
    writer.add_images('origin', imgs, i)
    writer.add_images('output', out, i)

writer.close()

