import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Xiong_Seq(nn.Module):
    def __init__(self):
        super(Xiong_Seq, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool6 = nn.MaxPool2d(2)
        self.flat7 = nn.Flatten()
        self.linear8 = nn.Linear(in_features=64*4*4, out_features=64)
        self.linear9 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool6(x)
        x = self.flat7(x)
        x = self.linear8(x)
        x = self.linear9(x)
        return x



net = Xiong_Seq()

input = torch.randn(64, 3, 32, 32)
out = net(input)
print(out.size())

writer = SummaryWriter("logs")
writer.add_graph(net, input)
writer.close()