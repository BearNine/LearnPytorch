import torch
import torch.utils
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn


#神经网络搭建
class Xiong(nn.Module):
    def __init__(self):
        super(Xiong, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    

if __name__ == '__main__':
    xiong = Xiong()
    input = torch.randn((64, 3, 32, 32))
    output = xiong(input)
    print(output.size())