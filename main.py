"""
ResNet 迁移学习 — CIFAR-10 图像分类
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import build_model
from src.data  import get_dataloaders, CLASSES
from src.train import run_training, save_metrics
from src.utils import plot_curves, plot_predictions

# 路径配置
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # # ① 加载数据
    # print("\n加载 CIFAR-10 数据...")
    # print("  如果下载失败，手动下载 cifar-10-python.tar.gz 放到 data/ 目录")
    # train_loader, test_loader, n_train, n_test = get_dataloaders(DATA_DIR)
    # print(f"  训练集: {n_train} 张, 测试集: {n_test} 张")
    # print(f"  类别: {', '.join(CLASSES)}")

    # # ② 创建模型（冻结 backbone，只训练分类头）
    # model = build_model(num_classes=10, freeze_backbone=False)
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total     = sum(p.numel() for p in model.parameters())
    # print(f"\n网络: ResNet-18 (pretrained)")
    # print(f"  总参数: {total:,}")
    # print(f"  可训练: {trainable:,} (仅全连接层)")
    # print(f"  冻结: {total - trainable:,} (卷积骨干)")

    # # ③ 训练
    # print("\n开始训练...")
    # train_losses, test_accs = run_training(
    #     model, train_loader, test_loader,
    #     epochs=20, lr=0.0001, device=device,
    #     model_dir=MODEL_DIR, checkpoint_every=5
    # )

    # # ④ 保存最终模型
    # final_path = os.path.join(MODEL_DIR, "resnet18_cifar10.pth")
    # torch.save(model.state_dict(), final_path)
    # print(f"\n训练完成，模型已保存: models/resnet18_cifar10.pth")
    # print(f"  最终准确率: {test_accs[-1]:.2f}%")
    
    # # ⑤ 持久化指标
    # metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    # save_metrics(train_losses, test_accs, metrics_path)
    # print(f"  指标已保存: models/metrics.json")


    # ③ 加载已训练模型
    model = build_model(num_classes=10, freeze_backbone=True)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "resnet18_cifar10.pth"),
        map_location=device
    ))
    model.to(device)
    print("\n已加载模型: models/resnet18_cifar10.pth")

    # 从 metrics.json 读取训练数据
    import json
    with open(os.path.join(MODEL_DIR, "metrics.json"), 'r') as f:
        m = json.load(f)
    train_losses, test_accs = m['train_losses'], m['test_accs']
    print(f"  最终准确率: {test_accs[-1]:.2f}%")


    # ⑥ 可视化
    print("\n生成可视化图表...")

    # 重新加载测试集（不包含训练时的增强，用于可视化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    test_data = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=False, transform=test_transform
    )

    plot_curves(train_losses, test_accs,
                os.path.join(MODEL_DIR, "loss_acc_curve.png"))
    print("  已保存: models/loss_acc_curve.png")

    plot_predictions(model, test_data, CLASSES,
                     os.path.join(MODEL_DIR, "predictions.png"),
                     device=device)
    print("  已保存: models/predictions.png")

    print("\n全部完成!")


if __name__ == "__main__":
    main()
