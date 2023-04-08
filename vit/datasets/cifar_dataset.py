# http://www.cs.toronto.edu/~kriz/cifar.html
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from vit.config import *
import os

# CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#预处理,自定义的train_transformer

transform= {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪为不同大小，默认0.08~1.0，期望输出大小224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，默认概率0.5
        transforms.ToTensor(),  # 转为tensor, 范围改为0~1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}


# 加载数据
def load_data(args):
    # CIFAR10数据集大小为32x32
    train_data = CIFAR100(root=args.data_path,
                          train=True,
                          download=False,
                          transform=transform["train"])
    test_data = CIFAR100(root=args.data_path,
                         train=False,
                         download=False,
                         transform=transform["val"])
    
    # 在转为tensor前,为PIL文件,可显示图片.即删除transform=transforms后才可显示
    # print(test_data[10])
    # test_data[10][0].show()  #展示图片

    # 数据加载
    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.batch_size)
    test_loader = DataLoader(test_data,
                             shuffle=True,
                             batch_size=args.batch_size)
    num_iter = len(train_loader)  # 50000 / 64 = 781.3 -> 782
    return train_loader, test_loader, num_iter


# if __name__ == '__main__':
    # train_loader, test_loader, num_iter = load_data(args)
