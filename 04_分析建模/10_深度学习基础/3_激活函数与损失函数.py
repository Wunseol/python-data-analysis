# 数据来源: 模拟数据
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === 激活函数 ===
print("=" * 50)
print("1. 常用激活函数")
print("=" * 50)

x = torch.linspace(-5, 5, 100)

# ReLU
relu_out = F.relu(x)
print(f"ReLU: max(0, x)")
print(f"  x=-2 → {F.relu(torch.tensor(-2.0)):.2f}")
print(f"  x=0  → {F.relu(torch.tensor(0.0)):.2f}")
print(f"  x=2  → {F.relu(torch.tensor(2.0)):.2f}")

# Sigmoid
sigmoid_out = torch.sigmoid(x)
print(f"\nSigmoid: 1/(1+e^(-x))")
print(f"  x=-2 → {torch.sigmoid(torch.tensor(-2.0)):.4f}")
print(f"  x=0  → {torch.sigmoid(torch.tensor(0.0)):.4f}")
print(f"  x=2  → {torch.sigmoid(torch.tensor(2.0)):.4f}")

# Tanh
tanh_out = torch.tanh(x)
print(f"\nTanh: (e^x - e^(-x))/(e^x + e^(-x))")
print(f"  x=-2 → {torch.tanh(torch.tensor(-2.0)):.4f}")
print(f"  x=0  → {torch.tanh(torch.tensor(0.0)):.4f}")
print(f"  x=2  → {torch.tanh(torch.tensor(2.0)):.4f}")

# Softmax
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"\nSoftmax:")
print(f"  输入logits: {logits.tolist()}")
print(f"  输出概率: {[f'{p:.4f}' for p in probs.tolist()]}")
print(f"  概率之和: {probs.sum():.4f}")

# === 激活函数梯度特性 ===
print("\n" + "=" * 50)
print("2. 激活函数梯度特性")
print("=" * 50)

x_grad = torch.linspace(-5, 5, 11)

# ReLU梯度
print("ReLU梯度: x>0时为1, x<=0时为0")
for val in [-3.0, -0.5, 0.0, 0.5, 3.0]:
    t = torch.tensor(val, requires_grad=True)
    y = F.relu(t)
    y.backward()
    print(f"  x={val:5.1f} → grad={t.grad.item():.1f}")

# Sigmoid梯度
print("\nSigmoid梯度: σ(x)(1-σ(x)), 两端梯度趋近0(梯度消失)")
for val in [-5.0, -2.0, 0.0, 2.0, 5.0]:
    t = torch.tensor(val, requires_grad=True)
    y = torch.sigmoid(t)
    y.backward()
    print(f"  x={val:5.1f} → σ={y.item():.4f}, grad={t.grad.item():.4f}")

# === 损失函数 ===
print("\n" + "=" * 50)
print("3. MSE损失函数(回归)")
print("=" * 50)

pred = torch.tensor([2.5, 0.0, 2.1, 7.8])
target = torch.tensor([3.0, -0.5, 2.0, 8.0])

mse_loss = nn.MSELoss()(pred, target)
print(f"预测值: {pred.tolist()}")
print(f"真实值: {target.tolist()}")
print(f"MSE损失: {mse_loss.item():.4f}")

# 手动计算验证
manual_mse = ((pred - target) ** 2).mean()
print(f"手动计算: {manual_mse.item():.4f}")

# === CrossEntropy损失函数(分类) ===
print("\n" + "=" * 50)
print("4. CrossEntropy损失函数(分类)")
print("=" * 50)

logits = torch.tensor([[1.5, 0.5, -0.5],
                        [0.1, 2.0, 0.3],
                        [-1.0, 0.5, 2.5]])
targets = torch.tensor([0, 1, 2])

ce_loss = nn.CrossEntropyLoss()(logits, targets)
print(f"Logits:\n{logits}")
print(f"真实类别: {targets.tolist()}")
print(f"CrossEntropy损失: {ce_loss.item():.4f}")

# 查看预测概率
probs = F.softmax(logits, dim=1)
print(f"预测概率:\n{probs.detach().round(decimals=4)}")
print(f"预测类别: {probs.argmax(dim=1).tolist()}")
print(f"正确预测: {(probs.argmax(dim=1) == targets).tolist()}")

# === 损失函数对比 ===
print("\n" + "=" * 50)
print("5. 损失函数选择指南")
print("=" * 50)

# 二分类
bce_logits = torch.tensor([2.0, -1.0, 0.5])
bce_targets = torch.tensor([1.0, 0.0, 1.0])
bce_loss = nn.BCEWithLogitsLoss()(bce_logits, bce_targets)
print(f"二分类BCE损失: {bce_loss.item():.4f}")

# 多分类
print(f"多分类CE损失: {ce_loss.item():.4f}")

# === 激活函数与损失函数搭配 ===
print("\n" + "=" * 50)
print("6. 常见搭配方式")
print("=" * 50)
print("回归任务:   输出层无激活 + MSELoss")
print("二分类:     输出层无激活 + BCEWithLogitsLoss")
print("多分类:     输出层无激活 + CrossEntropyLoss")
print("注意: PyTorch的CE/BCE已内置Softmax/Sigmoid，无需手动添加")
