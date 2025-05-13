import torchvision
from torch.utils.data import DataLoader
from torch.ustils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root=r"./cifdataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root=r"./cifdataset", train=False, transform=dataset_transform, download=True)

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("./logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch {}".format(epoch), imgs, step)
        step += 1
writer.close()


# with SummaryWriter("torchvision_logs") as writer:
#     for i in range(10):  # 展示前10个样本
#         img, target = train_set[i]
#         writer.add_image("train_set_samples", img, i)  # 修正标签名称
