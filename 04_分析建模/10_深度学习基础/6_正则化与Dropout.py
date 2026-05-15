# 数据来源: 模拟数据
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy

torch.manual_seed(42)
np.random.seed(42)

# === 生成容易过拟合的数据 ===
X, y = make_classification(
    n_samples=300, n_features=20, n_informative=5,
    n_redundant=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# === L2正则化(weight_decay) ===
print("=" * 50)
print("1. L2正则化(weight_decay)")
print("=" * 50)

class OverfitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 无正则化
model_no_reg = OverfitNet()
opt_no = optim.Adam(model_no_reg.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model_no_reg.train()
    for x, y in train_loader:
        opt_no.zero_grad()
        loss = criterion(model_no_reg(x), y)
        loss.backward()
        opt_no.step()

# L2正则化
model_l2 = OverfitNet()
opt_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(100):
    model_l2.train()
    for x, y in train_loader:
        opt_l2.zero_grad()
        loss = criterion(model_l2(x), y)
        loss.backward()
        opt_l2.step()

print(f"无正则化 → 训练: {evaluate(model_no_reg, train_loader):.2%}, 测试: {evaluate(model_no_reg, test_loader):.2%}")
print(f"L2正则化 → 训练: {evaluate(model_l2, train_loader):.2%}, 测试: {evaluate(model_l2, test_loader):.2%}")

# === Dropout ===
print("\n" + "=" * 50)
print("2. Dropout正则化")
print("=" * 50)

class DropoutNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model_drop = DropoutNet(dropout_rate=0.3)
opt_drop = optim.Adam(model_drop.parameters(), lr=0.001)

for epoch in range(100):
    model_drop.train()
    for x, y in train_loader:
        opt_drop.zero_grad()
        loss = criterion(model_drop(x), y)
        loss.backward()
        opt_drop.step()

print(f"Dropout(0.3) → 训练: {evaluate(model_drop, train_loader):.2%}, 测试: {evaluate(model_drop, test_loader):.2%}")

# === BatchNorm ===
print("\n" + "=" * 50)
print("3. BatchNorm批归一化")
print("=" * 50)

class BNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model_bn = BNNNet()
opt_bn = optim.Adam(model_bn.parameters(), lr=0.001)

for epoch in range(100):
    model_bn.train()
    for x, y in train_loader:
        opt_bn.zero_grad()
        loss = criterion(model_bn(x), y)
        loss.backward()
        opt_bn.step()

print(f"BatchNorm → 训练: {evaluate(model_bn, train_loader):.2%}, 测试: {evaluate(model_bn, test_loader):.2%}")

# === 早停法EarlyStopping ===
print("\n" + "=" * 50)
print("4. 早停法EarlyStopping")
print("=" * 50)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

model_es = OverfitNet()
opt_es = optim.Adam(model_es.parameters(), lr=0.001)
early_stop = EarlyStopping(patience=10)

for epoch in range(200):
    model_es.train()
    for x, y in train_loader:
        opt_es.zero_grad()
        loss = criterion(model_es(x), y)
        loss.backward()
        opt_es.step()

    model_es.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            val_loss += criterion(model_es(x), y).item()
    val_loss /= len(test_loader)

    if early_stop.should_stop:
        print(f"早停于Epoch {epoch+1}")
        break

model_es.load_state_dict(early_stop.best_model)
print(f"早停法 → 训练: {evaluate(model_es, train_loader):.2%}, 测试: {evaluate(model_es, test_loader):.2%}")

# === 过拟合对比 ===
print("\n" + "=" * 50)
print("5. 各方法过拟合对比")
print("=" * 50)

results = {
    "无正则化": (evaluate(model_no_reg, train_loader), evaluate(model_no_reg, test_loader)),
    "L2正则化": (evaluate(model_l2, train_loader), evaluate(model_l2, test_loader)),
    "Dropout": (evaluate(model_drop, train_loader), evaluate(model_drop, test_loader)),
    "BatchNorm": (evaluate(model_bn, train_loader), evaluate(model_bn, test_loader)),
    "早停法": (evaluate(model_es, train_loader), evaluate(model_es, test_loader)),
}

print(f"{'方法':>10} {'训练准确率':>10} {'测试准确率':>10} {'过拟合差距':>10}")
print("-" * 42)
for name, (train_acc, test_acc) in results.items():
    gap = train_acc - test_acc
    print(f"{name:>10} {train_acc:>10.2%} {test_acc:>10.2%} {gap:>10.2%}")
