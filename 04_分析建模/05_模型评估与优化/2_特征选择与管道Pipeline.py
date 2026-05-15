# 数据来源: sklearn.datasets.load_wine (红酒数据集)

"""
特征选择与管道 Pipeline
========================
本案例演示:
1. SelectKBest 特征选择
2. f_classif F检验特征选择
3. Pipeline 管道
4. ColumnTransformer 列变换器 (概念)
5. make_pipeline 快捷管道
6. 组合预处理与模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、数据加载与划分
# ============================================================

wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

print("=" * 60)
print("【红酒数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别分布: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ============================================================
# 二、SelectKBest 特征选择
# ============================================================

# 使用 F 检验选择最重要的 K 个特征
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print("\n" + "=" * 60)
print("【SelectKBest 特征选择 (K=5)】")
print(f"原始特征数: {X_train.shape[1]}")
print(f"选择后特征数: {X_train_selected.shape[1]}")

# 查看所有特征的 F 分数和 p 值
f_scores = selector.scores_
p_values = selector.pvalues_
selected_mask = selector.get_support()

print(f"\n{'特征名称':<25} {'F分数':>10} {'p值':>12} {'是否选中':>8}")
print("-" * 57)
for i, name in enumerate(feature_names):
    selected = "✓" if selected_mask[i] else "✗"
    print(f"{name:<25} {f_scores[i]:>10.2f} {p_values[i]:>12.6f} {selected:>8}")

print(f"\n选中的特征: {[feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]}")

# ============================================================
# 三、不同 K 值对模型性能的影响
# ============================================================

k_range = range(1, X_train.shape[1] + 1)
acc_list = []

for k in k_range:
    pipe_temp = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=k)),
        ('clf', LogisticRegression(max_iter=200, random_state=42))
    ])
    scores = cross_val_score(pipe_temp, X_train, y_train, cv=5, scoring='accuracy')
    acc_list.append(scores.mean())

best_k = list(k_range)[np.argmax(acc_list)]
print("\n" + "=" * 60)
print("【不同 K 值对模型性能的影响】")
print(f"{'K值':<6} {'交叉验证准确率':>16}")
print("-" * 24)
for k, acc in zip(k_range, acc_list):
    marker = " <-- 最优" if k == best_k else ""
    print(f"{k:<6} {acc:>16.4f}{marker}")

print(f"\n最优 K 值: {best_k}, 对应准确率: {max(acc_list):.4f}")

# ============================================================
# 四、Pipeline 管道
# ============================================================

# 手动构建 Pipeline (指定步骤名称)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=best_k)),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred_pipe = pipe.predict(X_test)

print("\n" + "=" * 60)
print("【Pipeline 管道】")
print(f"管道步骤:")
for name, step in pipe.named_steps.items():
    print(f"  {name}: {step.__class__.__name__}")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_pipe):.4f}")

# ============================================================
# 五、make_pipeline 快捷方式
# ============================================================

# make_pipeline 自动以类名作为步骤名称
pipe_quick = make_pipeline(
    StandardScaler(),
    SelectKBest(score_func=f_classif, k=best_k),
    LogisticRegression(max_iter=200, random_state=42)
)

pipe_quick.fit(X_train, y_train)
y_pred_quick = pipe_quick.predict(X_test)

print("\n" + "=" * 60)
print("【make_pipeline 快捷方式】")
print(f"管道步骤:")
for name, step in pipe_quick.named_steps.items():
    print(f"  {name}: {step.__class__.__name__}")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_quick):.4f}")

# ============================================================
# 六、ColumnTransformer 概念说明
# ============================================================

print("\n" + "=" * 60)
print("【ColumnTransformer 概念说明】")
print("ColumnTransformer 可对不同的列应用不同的变换:")
print("  from sklearn.compose import ColumnTransformer")
print("  ct = ColumnTransformer([")
print("      ('num', StandardScaler(), numerical_features),")
print("      ('cat', OneHotEncoder(), categorical_features)")
print("  ])")
print("在红酒数据集中所有特征都是数值型, 故直接使用 StandardScaler")
print("当数据集同时包含数值型和类别型特征时, ColumnTransformer 特别有用")

# ColumnTransformer 实际示例: 将前10个特征标准化, 后3个特征通过 SelectKBest
ct = ColumnTransformer([
    ('scale_all', StandardScaler(), list(range(X.shape[1])))
])

pipe_ct = Pipeline([
    ('preprocess', ct),
    ('select', SelectKBest(score_func=f_classif, k=best_k)),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])
pipe_ct.fit(X_train, y_train)
print(f"\nColumnTransformer + Pipeline 测试集准确率: {pipe_ct.score(X_test, y_test):.4f}")

# ============================================================
# 七、Pipeline + GridSearchCV 网格搜索
# ============================================================

pipe_gs = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif)),
    ('clf', LogisticRegression(max_iter=500, random_state=42))
])

param_grid = {
    'select__k': [3, 5, 8, 10, 13],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear']
}

grid_search = GridSearchCV(pipe_gs, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【Pipeline + GridSearchCV 网格搜索结果】")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

y_pred_gs = grid_search.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred_gs):.4f}")

print("\n【分类报告】")
print(classification_report(y_test, y_pred_gs, target_names=wine.target_names))

# ============================================================
# 八、对比: 有无特征选择
# ============================================================

# 无特征选择
pipe_no_select = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])
pipe_no_select.fit(X_train, y_train)
acc_no_select = pipe_no_select.score(X_test, y_test)

# 有特征选择
pipe_with_select = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=best_k)),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])
pipe_with_select.fit(X_train, y_train)
acc_with_select = pipe_with_select.score(X_test, y_test)

print("\n" + "=" * 60)
print("【有无特征选择对比】")
print(f"无特征选择 (全部{X.shape[1]}个特征): 准确率={acc_no_select:.4f}")
print(f"有特征选择 (选{best_k}个特征): 准确率={acc_with_select:.4f}")

# ============================================================
# 九、可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 特征 F 分数
sorted_idx = np.argsort(f_scores)[::-1]
colors = ['steelblue' if selected_mask[i] else 'lightgray' for i in sorted_idx]
axes[0].barh(range(len(f_scores)), f_scores[sorted_idx], color=colors)
axes[0].set_yticks(range(len(f_scores)))
axes[0].set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
axes[0].set_xlabel('F 分数', fontsize=12)
axes[0].set_title(f'特征 F 分数 (蓝色=选中, K={best_k})', fontsize=13)
axes[0].invert_yaxis()

# 2. K值对准确率的影响
axes[1].plot(k_range, acc_list, 'o-', color='steelblue', markersize=6)
axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
                label=f'最优K={best_k}')
axes[1].set_xlabel('选择的特征数 K', fontsize=12)
axes[1].set_ylabel('交叉验证准确率', fontsize=12)
axes[1].set_title('SelectKBest: K值对准确率的影响', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# 3. 有无特征选择对比
bars = axes[2].bar(['无特征选择\n(全部特征)', f'有特征选择\n(K={best_k})'],
                    [acc_no_select, acc_with_select],
                    color=['steelblue', 'darkorange'], alpha=0.8)
axes[2].set_ylabel('准确率', fontsize=12)
axes[2].set_title('特征选择效果对比', fontsize=13)
for bar, acc in zip(bars, [acc_no_select, acc_with_select]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', fontsize=12)
axes[2].set_ylim(0.9, 1.05)

plt.tight_layout()
plt.savefig('特征选择与Pipeline.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("特征选择与管道 Pipeline 演示完成!")
