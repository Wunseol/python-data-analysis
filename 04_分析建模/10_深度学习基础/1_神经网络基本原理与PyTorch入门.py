# 数据来源: 模拟数据
# 依赖库最低版本要求: torch>=2.0, numpy>=1.24
import torch
import numpy as np

# === PyTorch张量基础 ===
print("=" * 50)
print("1. PyTorch张量操作")
print("=" * 50)

# 创建张量
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.arange(0, 12).reshape(3, 4).float()
c = torch.randn(3, 3)

print(f"一维张量: {a}")
print(f"二维张量形状: {b.shape}")
print(f"随机张量:\n{c}")

# 张量运算
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(f"\n加法: {x + y}")
print(f"逐元素乘法: {x * y}")
print(f"点积: {torch.dot(x, y)}")
print(f"矩阵乘法: {torch.matmul(b, b.T).shape}")

# GPU支持
print(f"\nCUDA可用: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}")

# 张量与NumPy互转
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array).float()
np_from_tensor = a.numpy()
print(f"NumPy→Tensor: {tensor_from_np}")
print(f"Tensor→NumPy: {np_from_tensor}")

# === 自动求导autograd ===
print("\n" + "=" * 50)
print("2. 自动求导autograd")
print("=" * 50)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(f"y = x² + 3x + 1")
print(f"x = {x.item()}, y = {y.item()}")
print(f"dy/dx = 2x + 3 = {x.grad.item()} (理论值: {2*2+3})")

# 多变量求导
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x_input = torch.tensor([0.5, 1.0, 1.5])
output = torch.dot(w, x_input) + b
loss = output ** 2
loss.backward()
print(f"\nw梯度: {w.grad}")
print(f"b梯度: {b.grad}")

# 手动验证: loss = (w·x + b)², ∂loss/∂w = 2(w·x+b)·x
manual_w_grad = 2 * output.item() * x_input
print(f"手动验证w梯度: {manual_w_grad}")

# === 计算图概念 ===
print("\n" + "=" * 50)
print("3. 计算图概念")
print("=" * 50)

x = torch.tensor(3.0, requires_grad=True)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(1.5, requires_grad=True)

h = x * w1
y = h * w2
loss = (y - 10) ** 2

loss.backward()
print(f"x={x.item()}, w1={w1.item()}, w2={w2.item()}")
print(f"h = x * w1 = {h.item()}")
print(f"y = h * w2 = {y.item()}")
print(f"loss = (y - 10)² = {loss.item():.2f}")
print(f"dloss/dw2 = {w2.grad.item():.4f}")
print(f"dloss/dw1 = {w1.grad.item():.4f}")

# 梯度清零
w1.grad.zero_()
w2.grad.zero_()
print(f"梯度清零后: w1.grad={w1.grad}, w2.grad={w2.grad}")

# === nn.Module基础 ===
print("\n" + "=" * 50)
print("4. nn.Module基础")
print("=" * 50)

import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
print(f"模型结构:\n{model}")
print(f"\n参数列表:")
for name, param in model.named_parameters():
    print(f"  {name}: 形状{param.shape}")

# 前向传播
sample_input = torch.randn(2, 4)
output = model(sample_input)
print(f"\n输入形状: {sample_input.shape}")
print(f"输出形状: {output.shape}")
print(f"输出:\n{output.detach()}")
