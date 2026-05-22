# ResNet 迁移学习 — CIFAR-10 图像分类

使用 ImageNet 预训练的 ResNet-18 在 CIFAR-10 上进行迁移学习。

## 技术栈

- Python 3.12+
- PyTorch + torchvision
- matplotlib

## 快速开始

```bash
pip install torch torchvision matplotlib
python main.py
```

首次运行自动下载 CIFAR-10 数据集和 ResNet-18 预训练权重（约 45MB）。

## 项目结构

```
phase1c_resnet/
  README.md              项目说明
  main.py                 训练入口
  src/
    model.py              模型定义 (ResNet-18 + build_model)
    data.py               数据加载与预处理
    train.py              训练循环 + checkpoint/metrics 持久化
    utils.py              可视化工具
  data/                   数据集
  models/
    resnet18_cifar10.pth  最终模型
    checkpoint_epoch_*.pth 训练中间状态
    metrics.json           训练指标
    loss_acc_curve.png    损失与准确率曲线
    predictions.png       测试样本预测结果
```

## 迁移学习策略

`build_model(num_classes, freeze_backbone)` 支持两种模式：

| 参数 | 策略 | 说明 |
|------|------|------|
| `freeze_backbone=True` | 冻结骨干 | 只训练分类头，速度快但需额外解冻 BN |
| `freeze_backbone=False` | 全量微调 | 整网可训，效果最好但训练较慢 |

推荐全量微调（`freeze_backbone=False, lr=0.0001`），CIFAR-10 与 ImageNet 数据分布差异大，冻结骨干会导致 BN 统计量不匹配。

可选适配小图尺寸（32x32）：

```python
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
```

注意：改 conv1 后必须配合解冻骨干或全量微调，否则随机初始化的新 conv1 会破坏预训练特征流。

## 结果

| 策略 | 准确率 |
|------|--------|
| 冻结骨干 (仅 FC) | 40-42% |
| 冻结骨干 + 解冻 BN | 85.36% |
| 全量微调 (lr=0.0001) | 94.62% |
