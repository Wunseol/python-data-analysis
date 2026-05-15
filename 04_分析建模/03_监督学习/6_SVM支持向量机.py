# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集)

"""
SVM 支持向量机
===============
本案例演示:
1. SVC 支持向量分类器
2. 核函数类型 (linear, rbf, poly)
3. C 参数 (正则化强度)
4. gamma 参数 (RBF核的影响范围)
5. 决策边界可视化概念
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、数据加载与划分
# ============================================================

iris = load_iris()
X = iris.data
y = iris.target

print("=" * 60)
print("【鸢尾花数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"类别名称: {iris.target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# SVM 对特征尺度敏感, 建议进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 二、不同核函数对比
# ============================================================

kernels = ['linear', 'rbf', 'poly']
print("\n" + "=" * 60)
print("【不同核函数对比】")
print(f"{'核函数':<12} {'训练准确率':>12} {'测试准确率':>12} {'支持向量数':>12}")
print("-" * 50)

for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0, random_state=42)
    svm.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, svm.predict(X_train_scaled))
    te = accuracy_score(y_test, svm.predict(X_test_scaled))
    n_sv = svm.n_support_.sum()
    print(f"{kernel:<12} {tr:>12.4f} {te:>12.4f} {n_sv:>12}")

# ============================================================
# 三、C 参数的影响
# ============================================================

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
print("\n" + "=" * 60)
print("【C 参数对 RBF 核 SVM 的影响】")
print(f"{'C值':<10} {'训练准确率':>12} {'测试准确率':>12} {'支持向量数':>12}")
print("-" * 48)

train_acc_c = []
test_acc_c = []
for c in C_values:
    svm_c = SVC(kernel='rbf', C=c, random_state=42)
    svm_c.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, svm_c.predict(X_train_scaled))
    te = accuracy_score(y_test, svm_c.predict(X_test_scaled))
    n_sv = svm_c.n_support_.sum()
    train_acc_c.append(tr)
    test_acc_c.append(te)
    print(f"{c:<10} {tr:>12.4f} {te:>12.4f} {n_sv:>12}")

# ============================================================
# 四、gamma 参数的影响
# ============================================================

gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0]
print("\n" + "=" * 60)
print("【gamma 参数对 RBF 核 SVM 的影响】")
print(f"{'gamma值':<10} {'训练准确率':>12} {'测试准确率':>12} {'支持向量数':>12}")
print("-" * 48)

train_acc_g = []
test_acc_g = []
for g in gamma_values:
    svm_g = SVC(kernel='rbf', C=1.0, gamma=g, random_state=42)
    svm_g.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, svm_g.predict(X_train_scaled))
    te = accuracy_score(y_test, svm_g.predict(X_test_scaled))
    n_sv = svm_g.n_support_.sum()
    train_acc_g.append(tr)
    test_acc_g.append(te)
    print(f"{g:<10} {tr:>12.4f} {te:>12.4f} {n_sv:>12}")

# ============================================================
# 五、最优模型分类报告
# ============================================================

svm_best = SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=42)
svm_best.fit(X_train_scaled, y_train)
y_pred_best = svm_best.predict(X_test_scaled)

print("\n" + "=" * 60)
print("【最优 SVM 模型分类报告 (kernel=rbf, C=1.0, gamma=0.1)】")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

# ============================================================
# 六、决策边界可视化 (使用前两个特征)
# ============================================================

X_2d = X[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, stratify=y, random_state=42
)

scaler_2d = StandardScaler()
X_train_2d_s = scaler_2d.fit_transform(X_train_2d)
X_test_2d_s = scaler_2d.transform(X_test_2d)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, kernel in enumerate(kernels):
    svm_2d = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_2d.fit(X_train_2d_s, y_train_2d)
    acc = accuracy_score(y_test_2d, svm_2d.predict(X_test_2d_s))

    h = 0.02
    x_min, x_max = X_train_2d_s[:, 0].min() - 1, X_train_2d_s[:, 0].max() + 1
    y_min, y_max = X_train_2d_s[:, 1].min() - 1, X_train_2d_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    axes[idx].scatter(X_train_2d_s[:, 0], X_train_2d_s[:, 1], c=y_train_2d,
                      cmap=plt.cm.Set1, edgecolors='black', s=40, alpha=0.8)
    axes[idx].scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1],
                      s=100, linewidth=1.5, facecolors='none', edgecolors='k',
                      label='支持向量')
    axes[idx].set_xlabel(f'{iris.feature_names[0]} (标准化)', fontsize=11)
    axes[idx].set_ylabel(f'{iris.feature_names[1]} (标准化)', fontsize=11)
    axes[idx].set_title(f'核函数: {kernel}\n准确率: {acc:.4f}', fontsize=13)
    axes[idx].legend(fontsize=9, loc='best')

plt.suptitle('SVM 不同核函数的决策边界', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('SVM_决策边界.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、C 和 gamma 参数影响可视化
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(C_values, train_acc_c, 'o-', label='训练集', color='steelblue')
axes[0].plot(C_values, test_acc_c, 's-', label='测试集', color='darkorange')
axes[0].set_xscale('log')
axes[0].set_xlabel('C 值 (log scale)', fontsize=12)
axes[0].set_ylabel('准确率', fontsize=12)
axes[0].set_title('C 参数对准确率的影响 (RBF核)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(gamma_values, train_acc_g, 'o-', label='训练集', color='steelblue')
axes[1].plot(gamma_values, test_acc_g, 's-', label='测试集', color='darkorange')
axes[1].set_xscale('log')
axes[1].set_xlabel('gamma 值 (log scale)', fontsize=12)
axes[1].set_ylabel('准确率', fontsize=12)
axes[1].set_title('gamma 参数对准确率的影响 (RBF核)', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SVM_参数影响.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 八、决策边界可视化概念说明
# ============================================================

print("\n" + "=" * 60)
print("【决策边界可视化概念说明】")
print("SVM 通过寻找最大间隔超平面来分类:")
print("  - linear 核: 线性决策边界")
print("  - rbf 核: 非线性决策边界, 可处理复杂模式")
print("  - poly 核: 多项式决策边界, degree 控制多项式阶数")
print("C 参数: 控制对误分类的惩罚, C越大越不容忍误分类 (可能过拟合)")
print("gamma 参数: 控制RBF核的影响范围, gamma越大影响范围越小 (可能过拟合)")
print("支持向量: 距离决策边界最近的样本点, 它们决定了决策边界的位置")

print("\n" + "=" * 60)
print("SVM 支持向量机演示完成!")
