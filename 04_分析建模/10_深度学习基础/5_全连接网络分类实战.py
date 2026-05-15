# 数据来源: sklearn内置数据集
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# === 数据加载与预处理 ===
print("=" * 50)
print("1. 数据加载与预处理")
print("=" * 50)

data = load_breast_cancer()
X, y = data.data, data.target
print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
print(f"类别分布: 良性{(y==1).sum()}, 恶性{(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === DataLoader数据加载 ===
print("\n" + "=" * 50)
print("2. DataLoader数据加载")
print("=" * 50)

train_dataset = TensorDataset(
    torch.FloatTensor(X_train), torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test), torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

for batch_x, batch_y in train_loader:
    print(f"批次形状: X={batch_x.shape}, y={batch_y.shape}")
    break

# === 全连接网络构建 ===
print("\n" + "=" * 50)
print("3. 全连接网络构建")
print("=" * 50)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

model = Classifier(input_dim=30, hidden_dim=64, output_dim=2)
print(f"模型结构:\n{model}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params}")

# === 训练循环 ===
print("\n" + "=" * 50)
print("4. 训练循环")
print("=" * 50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        correct += (output.argmax(dim=1) == batch_y).sum().item()
        total += batch_x.size(0)

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1:3d}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2%}")

# === 测试评估 ===
print("\n" + "=" * 50)
print("5. 测试评估")
print("=" * 50)

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        output = model(batch_x)
        preds = output.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.tolist())

test_acc = correct / total
print(f"测试准确率: {test_acc:.2%}")

# 混淆矩阵
from sklearn.metrics import classification_report
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=['恶性', '良性']))

# === 不同网络结构对比 ===
print("=" * 50)
print("6. 不同网络结构对比")
print("=" * 50)

configs = [
    ("浅层(30→16→2)", [30, 16, 2]),
    ("中层(30→64→32→2)", [30, 64, 32, 2]),
    ("深层(30→128→64→32→2)", [30, 128, 64, 32, 2]),
]

for name, dims in configs:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    m = nn.Sequential(*layers)

    opt = optim.Adam(m.parameters(), lr=0.001)
    for epoch in range(20):
        m.train()
        for batch_x, batch_y in train_loader:
            opt.zero_grad()
            loss = criterion(m(batch_x), batch_y)
            loss.backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        preds = m(torch.FloatTensor(X_test)).argmax(dim=1)
        acc = (preds == torch.LongTensor(y_test)).float().mean().item()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name:>28}: 准确率={acc:.2%}, 参数量={n_params}")
