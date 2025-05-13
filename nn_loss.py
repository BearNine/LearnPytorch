import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_data = torchvision.datasets.CIFAR10(root='./cifdataset', train=True, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)

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
        self.soft = nn.Softmax(dim=1)

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
        x = self.soft(x)
        return x


loss_Cross = nn.CrossEntropyLoss()
loss_NLL = nn.NLLLoss()
net = Xiong_Seq()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

for epoch in range(50):
    for data in test_loader:
        imgs, labels = data
        outputs = net(imgs)
        result_loss = loss_NLL(torch.log(outputs), labels)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
    print('Epoch:', epoch, '\tLoss:', float(result_loss))
