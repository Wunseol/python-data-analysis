# 数据来源: 模拟数据
import torch
import torch.nn as nn
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

# === 手动实现前向传播 ===
print("=" * 50)
print("1. 手动实现前向传播")
print("=" * 50)

# 简单两层网络: 2→3→1
X = torch.tensor([[0.5, 0.8],
                   [0.2, 0.9],
                   [0.7, 0.3]])
y_true = torch.tensor([[1.0], [0.0], [1.0]])

W1 = torch.randn(2, 3)
b1 = torch.zeros(3)
W2 = torch.randn(3, 1)
b2 = torch.zeros(1)

# 前向传播
h = X @ W1 + b1
h_relu = torch.relu(h)
y_pred = h_relu @ W2 + b2

print(f"输入X:\n{X}")
print(f"隐藏层(线性): \n{h.detach().round(decimals=4)}")
print(f"隐藏层(ReLU): \n{h_relu.detach().round(decimals=4)}")
print(f"预测输出:\n{y_pred.detach().round(decimals=4)}")
print(f"真实标签:\n{y_true}")

# === 损失计算 ===
print("\n" + "=" * 50)
print("2. 损失计算")
print("=" * 50)

loss = nn.MSELoss()(y_pred, y_true)
print(f"MSE损失: {loss.item():.4f}")

# === 手动实现反向传播 ===
print("\n" + "=" * 50)
print("3. 手动实现反向传播")
print("=" * 50)

# MSE梯度: dloss/dy_pred = 2(y_pred - y_true) / n
n = y_true.shape[0]
dloss_dy = 2 * (y_pred - y_true) / n

# 输出层梯度
dloss_dW2 = h_relu.T @ dloss_dy
dloss_db2 = dloss_dy.sum(dim=0)

# ReLU梯度
dloss_dh_relu = dloss_dy @ W2.T
dloss_dh = dloss_dh_relu * (h > 0).float()

# 隐藏层梯度
dloss_dW1 = X.T @ dloss_dh
dloss_db1 = dloss_dh.sum(dim=0)

print(f"dL/dW2:\n{dloss_dW2.detach().round(decimals=4)}")
print(f"dL/db2: {dloss_db2.detach().round(decimals=4)}")
print(f"dL/dW1:\n{dloss_dW1.detach().round(decimals=4)}")
print(f"dL/db1: {dloss_db1.detach().round(decimals=4)}")

# === 梯度下降更新 ===
print("\n" + "=" * 50)
print("4. 梯度下降更新")
print("=" * 50)

lr = 0.01
print(f"学习率: {lr}")

W1_old = W1.clone()
W1 = W1 - lr * dloss_dW1
b1 = b1 - lr * dloss_db1
W2 = W2 - lr * dloss_dW2
b2 = b2 - lr * dloss_db2

# 更新后前向传播
h_new = X @ W1 + b1
h_relu_new = torch.relu(h_new)
y_pred_new = h_relu_new @ W2 + b2
loss_new = nn.MSELoss()(y_pred_new, y_true)

print(f"更新前损失: {loss.item():.4f}")
print(f"更新后损失: {loss_new.item():.4f}")

# === 与PyTorch自动求导对比 ===
print("\n" + "=" * 50)
print("5. PyTorch自动求导验证")
print("=" * 50)

torch.manual_seed(42)
W1_auto = torch.randn(2, 3, requires_grad=True)
b1_auto = torch.zeros(3, requires_grad=True)
W2_auto = torch.randn(3, 1, requires_grad=True)
b2_auto = torch.zeros(1, requires_grad=True)

h_auto = X @ W1_auto + b1_auto
h_relu_auto = torch.relu(h_auto)
y_pred_auto = h_relu_auto @ W2_auto + b2_auto
loss_auto = nn.MSELoss()(y_pred_auto, y_true)

loss_auto.backward()

print(f"手动计算 dL/dW2:\n{dloss_dW2.detach().round(decimals=4)}")
print(f"自动求导 dL/dW2:\n{W2_auto.grad.round(decimals=4)}")
print(f"\n结果一致: {torch.allclose(dloss_dW2.detach(), W2_auto.grad, atol=1e-3)}")

# === 完整训练循环 ===
print("\n" + "=" * 50)
print("6. 完整训练循环(手动梯度)")
print("=" * 50)

torch.manual_seed(42)
W1 = torch.randn(2, 3, requires_grad=True)
b1 = torch.zeros(3, requires_grad=True)
W2 = torch.randn(3, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=0.1)

for epoch in range(50):
    optimizer.zero_grad()
    h = X @ W1 + b1
    h_relu = torch.relu(h)
    y_pred = h_relu @ W2 + b2
    loss = nn.MSELoss()(y_pred, y_true)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

print(f"\n最终预测:\n{y_pred.detach().round(decimals=2)}")
print(f"真实标签:\n{y_true}")
