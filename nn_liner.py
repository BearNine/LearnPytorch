import torch
import torchvision
from torch import nn
from torchvision import transforms

test_data = torchvision.datasets.CIFAR10(root='./cifdataset', train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)

class Xiong_Liner(nn.Module):
    def __init__(self):
        super(Xiong_Liner, self).__init__()
        self.liner1 = nn.Linear(3 * 32 * 32, 100)
        self.liner2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.liner1(x)
        x = self.liner2(x)
        return x

xiong_liner = Xiong_Liner()

for data in test_loader:
    imgs, labels = data
    print(imgs.size())
    imgs = imgs.view(imgs.size(0), -1)
    print(imgs.size())
    out = xiong_liner(imgs)
    print(out.size())