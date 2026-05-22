"""
ResNet 迁移学习 — CIFAR-10 图像分类
"""
import torch.nn as nn
from torchvision import models


def build_model(num_classes=10, freeze_backbone=True):
    """
    构建 ResNet-18 迁移学习模型。

    freeze_backbone=True:  冻结卷积层，只训练最后的全连接层（更快）
    freeze_backbone=False: 全网络微调（准确率更高但更慢）
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()   # 禁用第一个池化层

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True   # 分类头从头学
            elif 'bn' in name:
                param.requires_grad = True   # BN 层适应新数据分布
            else:
                param.requires_grad = False  # 卷积核保持预训练权重

    # if freeze_backbone:
    #     # 冻结所有预训练参数
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     # 只解冻最后的全连接层
    #     for param in model.fc.parameters():
    #         param.requires_grad = True

    # 替换最后一层：原 1000 类 -> CIFAR-10 的 10 类
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
