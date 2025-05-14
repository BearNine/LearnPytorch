import torch
import torch.utils
import torchvision
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from model import *

# 准备数据集
# 利用compose实现原始数据的大小裁剪和totensor
imgtoTensor = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_data = torchvision.datasets.CIFAR10(r"./cifdataset", train=True, transform=imgtoTensor, download=True)
test_data = torchvision.datasets.CIFAR10(r"./cifdataset", train=False, transform=imgtoTensor, download=True)


# 数据集长度
train_data_len = len(train_data)
test_data_len = len(test_data)
print(f"训练集长度：{train_data_len}\n")
print(f"测试集长度：{test_data_len}\n")
print(f"训练图片大小：{train_data[0][0].size()}")
print(f"测试图片大小：{test_data[0][0].size()}")


# 利用dataloader加载数据
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)


# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建网络
xiong = Xiong()
if torch.cuda.is_available():
    xiong.cuda()


# 损失函数
loss_NLL= nn.NLLLoss()
if torch.cuda.is_available():
    loss_NLL.cuda()


# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(xiong.parameters(), lr=learning_rate)


# 设置训练网络参数
# 记录训练次数
total_train_step = 0
# 记录测试速度
total_test_step = 0
# 记录测试损失函数
total_test_loss = 0
# 训练次数
epoch = 20
# 添加tensorboard
writer = SummaryWriter(f"./logs")
start_time = time.time()


# 训练网络
for i in range(epoch):
    print(f"——————————第{i+1}轮训练开始———————————")

    # 训练步骤
    xiong.train()
    for data in train_data_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # 得到output输出
        output =  xiong(imgs)
        # 利用loss_NLL计算损失函数
        loss = loss_NLL(torch.log(output), targets)
        # 优化器设置
        # 先将优化器的梯度置0，在利用loss的反向传播计算梯度，再利用优化器更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每100次报告
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数:{total_train_step},Loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    xiong.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 确保模型的梯度为0，不干扰模型参数
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = xiong(imgs)
            loss = loss_NLL(torch.log(output), targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    total_test_step += 1
    print(f"整体测试集上损失函数为:{total_test_loss}")
    print(f"整体测试集上正确率为:{total_accuracy/test_data_len}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_len, total_test_step)
torch.save(xiong.state_dict(), f"xiong_{i}.pth")

end_time = time.time()
print("总时间为{}s".format(end_time - start_time))
writer.close()
    





