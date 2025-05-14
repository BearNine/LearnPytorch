import torch
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from model import *


img_path = r'D:\O\test\飞机.png'
img = Image.open(img_path)
img = img.convert('RGB')

data = torchvision.datasets.CIFAR10(root=r'./cifdataset', train=False, download=True)
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
toPlt = transforms.ToPILImage()

img_tensor = transform(img)
img = toPlt(img_tensor)
img_tensor = img_tensor.reshape(-1, 3, 32, 32)
print(img_tensor.size())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xiong = Xiong()
xiong.load_state_dict(torch.load(r'./xiong_19.pth'))
xiong.to(device)

print(f"可以预测{data.classes}")
output = xiong(img_tensor)
img.show()
type = output.argmax(dim=1).item()
print("图片是:{}".format(data.classes[type]))





