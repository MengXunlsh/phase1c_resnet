"""
CIFAR-10 数据加载与预处理
"""
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 标准化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def get_dataloaders(data_dir, batch_size=128):
    """返回 train_loader, test_loader, n_train, n_test"""

    # ResNet 需要更大输入尺寸，CIFAR-10 原生 32x32
    # 实际 ResNet 第一层 kernel=7 可以处理 32x32，但效果不如稍大尺寸
    # 这里保持 32x32，ResNet 能跑但推荐后续用 224x224（见注释）

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    train_data = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=train_transform
    )
    test_data = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=1000, shuffle=False)

    return train_loader, test_loader, len(train_data), len(test_data)
