import torch
from torchvision import transforms

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, -1, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float)
toPlt = transforms.ToPILImage()
img = toPlt(input)
img.show()