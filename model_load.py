
import torch
import torchvision

vgg16_load = torchvision.models.vgg16(weights='DEFAULT')
vgg16_load.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16_load)