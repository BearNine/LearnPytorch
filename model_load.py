import torch
import torchvision
from model import *
from torchvision import transforms



# vgg16_load = torchvision.models.vgg16(weights='DEFAULT')
# vgg16_load.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16_load)

imgtoTensor = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

test_data = torchvision.datasets.CIFAR10(root=r'./cifdataset', train=False, transform=imgtoTensor, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = Xiong()
x.load_state_dict(torch.load(r"D:\O\test\xiong_19.pth"))
x = x.to(device)

loss_NLL = nn.NLLLoss()

optimizer = torch.optim.SGD(x.parameters(), lr=0.001)

test_data_len = len(test_data)
total = 0

for data in test_loader:
    imgs, labels = data
    imgs = imgs.to(device)
    labels = labels.to(device)
    outputs = x(imgs)
    accuracy = (outputs.argmax(dim=1) == labels).sum()
    total += accuracy.item()
print(f"整体测试集上正确率为:{total/test_data_len}")
