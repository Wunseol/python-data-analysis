# 数据来源: sklearn.datasets.load_wine (红酒数据集)

"""
决策树与随机森林 — 红酒分类
============================
本案例演示:
1. DecisionTreeClassifier 决策树分类器
2. RandomForestClassifier 随机森林分类器
3. max_depth 对决策树的影响
4. n_estimators 对随机森林的影响
5. feature_importances_ 特征重要性
6. 决策树可视化概念
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、数据加载与划分
# ============================================================

wine = load_wine()
X = wine.data
y = wine.target

print("=" * 60)
print("【红酒数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"特征名称: {wine.feature_names}")
print(f"类别名称: {wine.target_names}")
print(f"类别分布: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ============================================================
# 二、决策树分类器
# ============================================================

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt = dt.predict(X_test)

print("\n" + "=" * 60)
print("【决策树分类器 (max_depth=3)】")
print(f"训练集准确率: {accuracy_score(y_train, y_train_pred_dt):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_test_pred_dt):.4f}")
print(f"树深度: {dt.get_depth()}")
print(f"叶子节点数: {dt.get_n_leaves()}")

# ============================================================
# 三、max_depth 对决策树的影响
# ============================================================

depths = range(1, 16)
train_acc_dt = []
test_acc_dt = []

for depth in depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train, y_train)
    train_acc_dt.append(accuracy_score(y_train, dt_temp.predict(X_train)))
    test_acc_dt.append(accuracy_score(y_test, dt_temp.predict(X_test)))

print("\n" + "=" * 60)
print("【max_depth 对决策树的影响】")
print(f"{'深度':<8} {'训练准确率':>12} {'测试准确率':>12}")
print("-" * 34)
for d, tr, te in zip(depths, train_acc_dt, test_acc_dt):
    marker = " <-- 最优" if te == max(test_acc_dt) else ""
    print(f"{d:<8} {tr:>12.4f} {te:>12.4f}{marker}")

best_depth = list(depths)[np.argmax(test_acc_dt)]
print(f"\n最优 max_depth: {best_depth}, 对应测试准确率: {max(test_acc_dt):.4f}")

# ============================================================
# 四、随机森林分类器
# ============================================================

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

print("\n" + "=" * 60)
print("【随机森林分类器 (n_estimators=100, max_depth=5)】")
print(f"训练集准确率: {accuracy_score(y_train, y_train_pred_rf):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_test_pred_rf):.4f}")

# ============================================================
# 五、n_estimators 对随机森林的影响
# ============================================================

n_estimators_list = [1, 5, 10, 20, 50, 100, 200, 500]
train_acc_rf = []
test_acc_rf = []

for n in n_estimators_list:
    rf_temp = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    rf_temp.fit(X_train, y_train)
    train_acc_rf.append(accuracy_score(y_train, rf_temp.predict(X_train)))
    test_acc_rf.append(accuracy_score(y_test, rf_temp.predict(X_test)))

print("\n" + "=" * 60)
print("【n_estimators 对随机森林的影响】")
print(f"{'树数量':<10} {'训练准确率':>12} {'测试准确率':>12}")
print("-" * 36)
for n, tr, te in zip(n_estimators_list, train_acc_rf, test_acc_rf):
    print(f"{n:<10} {tr:>12.4f} {te:>12.4f}")

# ============================================================
# 六、特征重要性对比
# ============================================================

dt_full = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_full.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【特征重要性对比】")
print(f"{'特征名称':<25} {'决策树':>10} {'随机森林':>10}")
print("-" * 47)
for i, name in enumerate(wine.feature_names):
    print(f"{name:<25} {dt_full.feature_importances_[i]:>10.4f} {rf.feature_importances_[i]:>10.4f}")

# ============================================================
# 七、分类报告
# ============================================================

print("\n" + "=" * 60)
print("【决策树分类报告】")
print(classification_report(y_test, y_test_pred_dt, target_names=wine.target_names))

print("=" * 60)
print("【随机森林分类报告】")
print(classification_report(y_test, y_test_pred_rf, target_names=wine.target_names))

# ============================================================
# 八、可视化
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. max_depth 对决策树的影响
axes[0, 0].plot(depths, train_acc_dt, 'o-', label='训练集', color='steelblue')
axes[0, 0].plot(depths, test_acc_dt, 's-', label='测试集', color='darkorange')
axes[0, 0].axvline(x=best_depth, color='red', linestyle='--', alpha=0.7, label=f'最优深度={best_depth}')
axes[0, 0].set_xlabel('max_depth', fontsize=12)
axes[0, 0].set_ylabel('准确率', fontsize=12)
axes[0, 0].set_title('决策树: max_depth 对准确率的影响', fontsize=13)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. n_estimators 对随机森林的影响
axes[0, 1].plot(n_estimators_list, train_acc_rf, 'o-', label='训练集', color='steelblue')
axes[0, 1].plot(n_estimators_list, test_acc_rf, 's-', label='测试集', color='darkorange')
axes[0, 1].set_xlabel('n_estimators', fontsize=12)
axes[0, 1].set_ylabel('准确率', fontsize=12)
axes[0, 1].set_title('随机森林: n_estimators 对准确率的影响', fontsize=13)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# 3. 决策树特征重要性
dt_importance = dt_full.feature_importances_
sorted_idx_dt = np.argsort(dt_importance)[::-1]
axes[1, 0].barh(range(len(dt_importance)), dt_importance[sorted_idx_dt], color='steelblue')
axes[1, 0].set_yticks(range(len(dt_importance)))
axes[1, 0].set_yticklabels([wine.feature_names[i] for i in sorted_idx_dt], fontsize=9)
axes[1, 0].set_xlabel('重要性', fontsize=12)
axes[1, 0].set_title('决策树特征重要性', fontsize=13)
axes[1, 0].invert_yaxis()

# 4. 随机森林特征重要性
rf_importance = rf.feature_importances_
sorted_idx_rf = np.argsort(rf_importance)[::-1]
axes[1, 1].barh(range(len(rf_importance)), rf_importance[sorted_idx_rf], color='darkorange')
axes[1, 1].set_yticks(range(len(rf_importance)))
axes[1, 1].set_yticklabels([wine.feature_names[i] for i in sorted_idx_rf], fontsize=9)
axes[1, 1].set_xlabel('重要性', fontsize=12)
axes[1, 1].set_title('随机森林特征重要性', fontsize=13)
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('决策树与随机森林.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 九、决策树可视化概念说明
# ============================================================

print("\n" + "=" * 60)
print("【决策树可视化说明】")
print("sklearn 提供 export_text 和 plot_tree 函数用于可视化决策树:")
print("  from sklearn.tree import export_text, plot_tree")
print("  export_text(dt) — 输出文本形式的决策规则")
print("  plot_tree(dt) — 绘制决策树图形")
print("  也可使用 graphviz 库: export_graphviz(dt, out_file='tree.dot')")
print("注意: 随机森林由多棵决策树组成, 通常只可视化单棵树来理解结构")

print("\n" + "=" * 60)
print("决策树与随机森林演示完成!")
