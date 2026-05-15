# 数据来源: MNIST(sklearn digits备用)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# === 数据加载 ===
print("=" * 50)
print("1. 数据加载")
print("=" * 50)

try:
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    input_dim = 28 * 28
    num_classes = 10
    data_source = "MNIST"
    print(f"数据集: MNIST")
except Exception:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    input_dim = 64
    num_classes = 10
    data_source = "sklearn digits"
    print(f"数据集: sklearn digits (MNIST下载失败，使用备用)")

print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

# === 数据预处理 ===
print("\n" + "=" * 50)
print("2. 数据预处理与查看")
print("=" * 50)

if data_source == "MNIST":
    images, labels = next(iter(train_loader))
    print(f"批次形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"像素范围: [{images.min():.2f}, {images.max():.2f}]")
    print(f"样本标签: {labels[:10].tolist()}")
else:
    for batch_x, batch_y in train_loader:
        print(f"批次形状: {batch_x.shape}")
        print(f"标签: {batch_y[:10].tolist()}")
        break

# === 网络设计 ===
print("\n" + "=" * 50)
print("3. 网络设计")
print("=" * 50)

class DigitNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

model = DigitNet(input_dim, num_classes)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型结构:\n{model}")
print(f"总参数量: {total_params:,}")

# === 训练 ===
print("\n" + "=" * 50)
print("4. 训练")
print("=" * 50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        if data_source == "MNIST":
            images, labels = batch
            images = images.view(images.size(0), -1)
        else:
            images, labels = batch

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    avg_loss = total_loss / total
    train_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2%}")

# === 评估 ===
print("\n" + "=" * 50)
print("5. 测试评估")
print("=" * 50)

model.eval()
correct = 0
total = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes

with torch.no_grad():
    for batch in test_loader:
        if data_source == "MNIST":
            images, labels = batch
            images = images.view(images.size(0), -1)
        else:
            images, labels = batch

        output = model(images)
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if preds[i] == label:
                class_correct[label] += 1

test_acc = correct / total
print(f"测试准确率: {test_acc:.2%}")
print(f"\n各类别准确率:")
for i in range(num_classes):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        print(f"  数字{i}: {acc:.2%} ({class_correct[i]}/{class_total[i]})")

# === 预测可视化 ===
print("\n" + "=" * 50)
print("6. 预测结果展示")
print("=" * 50)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        if data_source == "MNIST":
            images, labels = batch
            flat_images = images.view(images.size(0), -1)
        else:
            flat_images, labels = batch
            images = flat_images

        output = model(flat_images)
        probs = torch.softmax(output, dim=1)
        preds = output.argmax(dim=1)
        confs = probs.max(dim=1).values

        print(f"{'索引':>4} {'真实':>4} {'预测':>4} {'置信度':>8} {'结果':>6}")
        print("-" * 30)
        for i in range(min(15, labels.size(0))):
            result = "✓" if preds[i] == labels[i] else "✗"
            print(f"{i:>4} {labels[i].item():>4} {preds[i].item():>4} {confs[i].item():>8.2%} {result:>6}")
        break

# 错误样本分析
print("\n错误样本:")
errors = (preds != labels).nonzero(as_tuple=True)[0]
for idx in errors[:5]:
    print(f"  真实={labels[idx].item()}, 预测={preds[idx].item()}, 置信度={confs[idx].item():.2%}")
