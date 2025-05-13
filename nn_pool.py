import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_data = torchvision.datasets.CIFAR10(root='./cifdataset', train=False, transform=transforms.ToTensor(), download=True)
test_data = torch.utils.data.DataLoader(test_data, batch_size=1)

class Xiong_Pool(nn.Module):
    def __init__(self):
        super(Xiong_Pool, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), ceil_mode=False)

    def forward(self, x):
        x = self.pool1(x)
        return x


xiong_chi = Xiong_Pool()

print(input)
out = xiong_chi(input)
print(out)

writer = SummaryWriter('logs')
for i, datajuan in enumerate(test_data):
    imgs, targets = datajuan
    out = xiong_chi(imgs)
    print(out.shape)
    # out = torch.reshape(out, (-1, 3, 30, 30))
    writer.add_images('origin', imgs, i)
    writer.add_images('output', out, i)

writer.close()