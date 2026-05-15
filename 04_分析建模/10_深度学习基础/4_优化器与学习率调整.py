# 数据来源: 模拟数据
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# === 生成模拟数据 ===
from sklearn.datasets import make_moons
X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
X = torch.FloatTensor(X_np)
y = torch.LongTensor(y_np)

# === 优化器对比 ===
print("=" * 50)
print("1. SGD / Adam / RMSprop 优化器对比")
print("=" * 50)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_model(optimizer_class, lr, **kwargs):
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=lr, **kwargs)
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # 计算准确率
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return losses, acc

optimizers = {
    'SGD(lr=0.1)': (optim.SGD, {'lr': 0.1}),
    'SGD+Momentum': (optim.SGD, {'lr': 0.1, 'momentum': 0.9}),
    'Adam(lr=0.01)': (optim.Adam, {'lr': 0.01}),
    'RMSprop(lr=0.01)': (optim.RMSprop, {'lr': 0.01}),
}

print(f"{'优化器':>18} {'最终损失':>10} {'准确率':>8}")
print("-" * 38)
for name, (opt_cls, params) in optimizers.items():
    losses, acc = train_model(opt_cls, **params)
    print(f"{name:>18} {losses[-1]:>10.4f} {acc:>8.2%}")

# === 学习率调度 ===
print("\n" + "=" * 50)
print("2. 学习率调度策略")
print("=" * 50)

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# StepLR: 每step_size个epoch，lr乘以gamma
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("--- StepLR (每20epoch衰减50%) ---")
lrs_step = []
for epoch in range(60):
    lrs_step.append(optimizer.param_groups[0]['lr'])
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    scheduler_step.step()
print(f"初始LR: {lrs_step[0]:.6f}")
print(f"20epoch后: {lrs_step[20]:.6f}")
print(f"40epoch后: {lrs_step[40]:.6f}")
print(f"60epoch后: {lrs_step[59]:.6f}")

# CosineAnnealing: 余弦退火
model2 = SimpleNet()
optimizer2 = optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)

print("\n--- CosineAnnealing (T_max=50) ---")
lrs_cos = []
for epoch in range(50):
    lrs_cos.append(optimizer2.param_groups[0]['lr'])
    optimizer2.zero_grad()
    output = model2(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer2.step()
    scheduler_cos.step()
print(f"初始LR: {lrs_cos[0]:.6f}")
print(f"25epoch: {lrs_cos[25]:.6f}")
print(f"50epoch: {lrs_cos[49]:.6f}")

# === 梯度裁剪 ===
print("\n" + "=" * 50)
print("3. 梯度裁剪")
print("=" * 50)

model3 = SimpleNet()
optimizer3 = optim.Adam(model3.parameters(), lr=0.01)

optimizer3.zero_grad()
output = model3(X)
loss = criterion(output, y)
loss.backward()

# 裁剪前梯度范数
total_norm_before = 0
for p in model3.parameters():
    if p.grad is not None:
        total_norm_before += p.grad.data.norm(2).item() ** 2
total_norm_before = total_norm_before ** 0.5

# 梯度裁剪
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model3.parameters(), max_norm=max_norm)

# 裁剪后梯度范数
total_norm_after = 0
for p in model3.parameters():
    if p.grad is not None:
        total_norm_after += p.grad.data.norm(2).item() ** 2
total_norm_after = total_norm_after ** 0.5

print(f"裁剪前梯度范数: {total_norm_before:.4f}")
print(f"裁剪后梯度范数: {total_norm_after:.4f}")
print(f"最大范数限制: {max_norm}")
print(f"是否被裁剪: {'是' if total_norm_before > max_norm else '否'}")

# === 综合训练: 优化器+调度器+裁剪 ===
print("\n" + "=" * 50)
print("4. 综合训练示例")
print("=" * 50)

model4 = SimpleNet()
optimizer4 = optim.Adam(model4.parameters(), lr=0.01)
scheduler4 = optim.lr_scheduler.StepLR(optimizer4, step_size=20, gamma=0.5)

for epoch in range(30):
    optimizer4.zero_grad()
    output = model4(X)
    loss = criterion(output, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model4.parameters(), max_norm=5.0)
    optimizer4.step()
    scheduler4.step()

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            preds = model4(X).argmax(dim=1)
            acc = (preds == y).float().mean().item()
        lr = optimizer4.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}, Acc: {acc:.2%}, LR: {lr:.6f}")
