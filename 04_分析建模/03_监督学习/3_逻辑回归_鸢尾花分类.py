# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集)

"""
逻辑回归 — 鸢尾花分类
======================
本案例演示:
1. LogisticRegression 逻辑回归模型
2. predict / predict_proba 预测方法
3. confusion_matrix 混淆矩阵
4. classification_report 分类报告
5. accuracy_score 准确率
6. 多分类问题处理
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

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
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")
print(f"类别分布: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ============================================================
# 二、构建与训练逻辑回归模型
# ============================================================

# multi_class='multinomial' 使用 softmax 进行多分类
# max_iter 增大以保证收敛
# C 是正则化强度的倒数, 值越小正则化越强
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    max_iter=200,
    random_state=42
)
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【模型参数】")
print(f"截距: {model.intercept_}")
print(f"系数矩阵形状: {model.coef_.shape}")
print(f"类别: {model.classes_}")
print(f"迭代次数: {model.n_iter_}")

# ============================================================
# 三、模型预测
# ============================================================

# predict — 返回预测类别
y_pred = model.predict(X_test)

# predict_proba — 返回每个样本属于各类别的概率
y_proba = model.predict_proba(X_test)

print("\n" + "=" * 60)
print("【预测结果示例 (前10个样本)】")
print(f"{'真实类别':<10} {'预测类别':<10} {'预测概率(setosa)':<18} {'预测概率(versicolor)':<22} {'预测概率(virginica)'}")
print("-" * 90)
for i in range(10):
    print(f"{y_test[i]:<10} {y_pred[i]:<10} {y_proba[i][0]:<18.4f} {y_proba[i][1]:<22.4f} {y_proba[i][2]:.4f}")

# ============================================================
# 四、模型评估
# ============================================================

# 准确率
acc_train = accuracy_score(y_train, model.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("【准确率】")
print(f"训练集准确率: {acc_train:.4f}")
print(f"测试集准确率: {acc_test:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n" + "=" * 60)
print("【混淆矩阵】")
print(f"{'':>20} {'预测setosa':>12} {'预测versicolor':>16} {'预测virginica':>14}")
for i, cls_name in enumerate(iris.target_names):
    print(f"{'真实'+cls_name:>20} {cm[i][0]:>12} {cm[i][1]:>16} {cm[i][2]:>14}")

# 分类报告
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\n" + "=" * 60)
print("【分类报告】")
print(report)

# ============================================================
# 五、不同 C 值对模型的影响
# ============================================================

C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
print("=" * 60)
print("【不同 C 值对模型的影响】")
print(f"{'C值':<10} {'训练准确率':>12} {'测试准确率':>12}")
print("-" * 36)
for c in C_values:
    model_c = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        C=c, max_iter=200, random_state=42
    )
    model_c.fit(X_train, y_train)
    acc_tr = accuracy_score(y_train, model_c.predict(X_train))
    acc_te = accuracy_score(y_test, model_c.predict(X_test))
    print(f"{c:<10} {acc_tr:>12.4f} {acc_te:>12.4f}")

# ============================================================
# 六、混淆矩阵可视化
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=iris.target_names,
       yticklabels=iris.target_names,
       title=f'混淆矩阵 (测试集准确率: {acc_test:.4f})',
       xlabel='预测类别',
       ylabel='真实类别')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=16)

plt.tight_layout()
plt.savefig('逻辑回归_混淆矩阵.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、决策边界可视化 (使用前两个特征)
# ============================================================

X_2d = X[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, stratify=y, random_state=42
)

model_2d = LogisticRegression(
    multi_class='multinomial', solver='lbfgs',
    C=1.0, max_iter=200, random_state=42
)
model_2d.fit(X_train_2d, y_train_2d)

h = 0.02
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 7))
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1,
                     edgecolors='black', s=50, alpha=0.8)
ax.set_xlabel(iris.feature_names[0], fontsize=12)
ax.set_ylabel(iris.feature_names[1], fontsize=12)
ax.set_title('逻辑回归决策边界 (前两个特征)', fontsize=14)
legend = ax.legend(*scatter.legend_elements(), title="类别",
                   loc="best", fontsize=10)
ax.add_artist(legend)
plt.tight_layout()
plt.savefig('逻辑回归_决策边界.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("逻辑回归 — 鸢尾花分类演示完成!")
