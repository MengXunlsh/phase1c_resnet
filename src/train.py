"""
训练与评估循环（含 checkpoint 保存）
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim


def train_epoch(model, loader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total


def save_checkpoint(model, optimizer, epoch, loss, acc, path):
    """保存训练中间状态"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
    }, path)


def save_metrics(train_losses, test_accs, path):
    """保存训练指标为 JSON"""
    with open(path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'test_accs': test_accs
        }, f, indent=2)


def run_training(model, train_loader, test_loader, epochs=20,
                 lr=0.001, device='cpu', model_dir='.',
                 checkpoint_every=5, verbose=True):
    """
    完整训练流程。
    checkpoint_every: 每隔多少 epoch 保存一次 checkpoint
    返回 (train_losses, test_accs)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    model.to(device)
    train_losses = []
    test_accs    = []

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc  = evaluate(model, test_loader, device)

        train_losses.append(loss)
        test_accs.append(acc)
        scheduler.step()

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:2d}: Loss={loss:.4f}, "
                  f"Test Acc={acc:.2f}%, LR={current_lr:.4f}")

        # 定期保存 checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = f"{model_dir}/checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch+1, loss, acc, ckpt_path)

    return train_losses, test_accs
