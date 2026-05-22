"""
可视化工具
"""
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def plot_curves(train_losses, test_accs, save_path):
    """绘制 Loss 和 Accuracy 曲线"""
    epochs = len(train_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs+1), train_losses, marker='o',
             color='#2c7fb8', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, epochs+1), test_accs, marker='s',
             color='#e31a1c', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(model, test_data, class_names, save_path, device='cpu'):
    """绘制 10 张预测样本"""
    loader = DataLoader(test_data, batch_size=10, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    for i in range(10):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * std + mean
        img = img.clip(0, 1)

        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        color = '#2ca02c' if true_label == pred_label else '#d62728'

        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_label}  Pred: {pred_label}',
                          color=color, fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
