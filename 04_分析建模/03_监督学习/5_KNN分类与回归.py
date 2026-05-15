# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集)

"""
KNN 分类与回归
===============
本案例演示:
1. KNeighborsClassifier KNN分类器
2. n_neighbors (K值) 的选择
3. 距离度量方式
4. KNeighborsRegressor KNN回归器
5. K值对准确率的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

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
print(f"类别分布: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ============================================================
# 二、KNN 分类器
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print("\n" + "=" * 60)
print("【KNN 分类器 (K=5)】")
print(f"训练集准确率: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_test_pred):.4f}")

# predict_proba — 返回各类别的概率
y_proba = knn.predict_proba(X_test)
print(f"\n前5个样本的预测概率:")
for i in range(5):
    print(f"  样本{i}: 真实={y_test[i]}, 预测={y_test_pred[i]}, 概率={y_proba[i]}")

# ============================================================
# 三、K值对准确率的影响
# ============================================================

k_range = range(1, 31)
train_acc_list = []
test_acc_list = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    train_acc_list.append(accuracy_score(y_train, knn_temp.predict(X_train)))
    test_acc_list.append(accuracy_score(y_test, knn_temp.predict(X_test)))

best_k = list(k_range)[np.argmax(test_acc_list)]
print("\n" + "=" * 60)
print("【K值对准确率的影响】")
print(f"{'K值':<6} {'训练准确率':>12} {'测试准确率':>12}")
print("-" * 32)
for k, tr, te in zip(k_range, train_acc_list, test_acc_list):
    marker = " <-- 最优" if te == max(test_acc_list) and k == best_k else ""
    print(f"{k:<6} {tr:>12.4f} {te:>12.4f}{marker}")

print(f"\n最优 K 值: {best_k}, 对应测试准确率: {max(test_acc_list):.4f}")

# ============================================================
# 四、距离度量方式对比
# ============================================================

metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
print("\n" + "=" * 60)
print("【不同距离度量方式对比 (K=5)】")
print(f"{'距离度量':<15} {'训练准确率':>12} {'测试准确率':>12}")
print("-" * 41)
for metric in metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_metric.fit(X_train, y_train)
    tr = accuracy_score(y_train, knn_metric.predict(X_train))
    te = accuracy_score(y_test, knn_metric.predict(X_test))
    print(f"{metric:<15} {tr:>12.4f} {te:>12.4f}")

# ============================================================
# 五、权重方式对比
# ============================================================

weights_list = ['uniform', 'distance']
print("\n" + "=" * 60)
print("【不同权重方式对比 (K=5)】")
print(f"{'权重方式':<15} {'训练准确率':>12} {'测试准确率':>12}")
print("-" * 41)
for w in weights_list:
    knn_w = KNeighborsClassifier(n_neighbors=5, weights=w)
    knn_w.fit(X_train, y_train)
    tr = accuracy_score(y_train, knn_w.predict(X_train))
    te = accuracy_score(y_test, knn_w.predict(X_test))
    print(f"{w:<15} {tr:>12.4f} {te:>12.4f}")

# ============================================================
# 六、KNN 回归器
# ============================================================

# 使用鸢尾花的第4个特征 (petal width) 预测第3个特征 (petal length)
X_reg = iris.data[:, 3:4]
y_reg = iris.data[:, 2]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_reg, y_train_reg)

y_train_reg_pred = knn_reg.predict(X_train_reg)
y_test_reg_pred = knn_reg.predict(X_test_reg)

print("\n" + "=" * 60)
print("【KNN 回归器 (K=5)】")
print(f"特征: petal width → 目标: petal length")
print(f"训练集 R²: {knn_reg.score(X_train_reg, y_train_reg):.4f}")
print(f"测试集 R²: {knn_reg.score(X_test_reg, y_test_reg):.4f}")
print(f"训练集 MSE: {mean_squared_error(y_train_reg, y_train_reg_pred):.4f}")
print(f"测试集 MSE: {mean_squared_error(y_test_reg, y_test_reg_pred):.4f}")

# K值对回归性能的影响
k_reg_range = range(1, 31)
test_r2_list = []
for k in k_reg_range:
    knn_reg_temp = KNeighborsRegressor(n_neighbors=k)
    knn_reg_temp.fit(X_train_reg, y_train_reg)
    test_r2_list.append(knn_reg_temp.score(X_test_reg, y_test_reg))

best_k_reg = list(k_reg_range)[np.argmax(test_r2_list)]
print(f"\nKNN回归最优 K 值: {best_k_reg}, 对应 R²: {max(test_r2_list):.4f}")

# ============================================================
# 七、分类报告
# ============================================================

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_test_pred_best = knn_best.predict(X_test)

print("\n" + "=" * 60)
print(f"【最优 K={best_k} 的分类报告】")
print(classification_report(y_test, y_test_pred_best, target_names=iris.target_names))

# ============================================================
# 八、可视化
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. K值对分类准确率的影响
axes[0, 0].plot(k_range, train_acc_list, 'o-', label='训练集', color='steelblue', markersize=4)
axes[0, 0].plot(k_range, test_acc_list, 's-', label='测试集', color='darkorange', markersize=4)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最优K={best_k}')
axes[0, 0].set_xlabel('K值', fontsize=12)
axes[0, 0].set_ylabel('准确率', fontsize=12)
axes[0, 0].set_title('KNN分类: K值对准确率的影响', fontsize=13)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. K值对回归R²的影响
axes[0, 1].plot(k_reg_range, test_r2_list, 's-', color='darkorange', markersize=4)
axes[0, 1].axvline(x=best_k_reg, color='red', linestyle='--', alpha=0.7, label=f'最优K={best_k_reg}')
axes[0, 1].set_xlabel('K值', fontsize=12)
axes[0, 1].set_ylabel('R²', fontsize=12)
axes[0, 1].set_title('KNN回归: K值对R²的影响', fontsize=13)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# 3. KNN回归拟合曲线
X_plot = np.linspace(X_reg.min(), X_reg.max(), 200).reshape(-1, 1)
knn_reg_best = KNeighborsRegressor(n_neighbors=best_k_reg)
knn_reg_best.fit(X_train_reg, y_train_reg)
y_plot = knn_reg_best.predict(X_plot)

axes[1, 0].scatter(X_train_reg, y_train_reg, alpha=0.5, s=20, color='steelblue', label='训练数据')
axes[1, 0].scatter(X_test_reg, y_test_reg, alpha=0.5, s=20, color='darkorange', label='测试数据')
axes[1, 0].plot(X_plot, y_plot, 'r-', linewidth=2, label=f'KNN回归 (K={best_k_reg})')
axes[1, 0].set_xlabel('petal width (cm)', fontsize=12)
axes[1, 0].set_ylabel('petal length (cm)', fontsize=12)
axes[1, 0].set_title('KNN回归拟合曲线', fontsize=13)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 4. 不同K值的决策边界 (前两个特征)
X_2d = X[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, stratify=y, random_state=42
)

for k_val in [1, 5, 15]:
    knn_2d = KNeighborsClassifier(n_neighbors=k_val)
    knn_2d.fit(X_train_2d, y_train_2d)
    acc = accuracy_score(y_test_2d, knn_2d.predict(X_test_2d))

    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[1, 1].contour(xx, yy, Z, alpha=0.3, levels=[0.5, 1.5, 2.5])

axes[1, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1,
                    edgecolors='black', s=30, alpha=0.8)
axes[1, 1].set_xlabel(iris.feature_names[0], fontsize=12)
axes[1, 1].set_ylabel(iris.feature_names[1], fontsize=12)
axes[1, 1].set_title('KNN决策边界 (K=1,5,15)', fontsize=13)

plt.tight_layout()
plt.savefig('KNN分类与回归.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("KNN 分类与回归演示完成!")
