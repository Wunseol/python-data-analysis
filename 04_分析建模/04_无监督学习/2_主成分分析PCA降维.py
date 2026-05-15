# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集)

"""
主成分分析 PCA 降维
====================
本案例演示:
1. PCA 主成分分析
2. n_components 主成分数量
3. explained_variance_ratio_ 方差解释比例
4. 累积方差解释率
5. 2D 降维可视化
6. StandardScaler 标准化在 PCA 前的重要性
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、数据加载
# ============================================================

iris = load_iris()
X = iris.data
y = iris.target

print("=" * 60)
print("【鸢尾花数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")
print(f"特征均值: {X.mean(axis=0)}")
print(f"特征标准差: {X.std(axis=0)}")

# ============================================================
# 二、不标准化的 PCA
# ============================================================

pca_raw = PCA()
X_pca_raw = pca_raw.fit_transform(X)

print("\n" + "=" * 60)
print("【不标准化的 PCA 结果】")
print(f"各主成分方差解释比例: {pca_raw.explained_variance_ratio_}")
print(f"累积方差解释比例: {np.cumsum(pca_raw.explained_variance_ratio_)}")
print(f"各主成分特征值: {pca_raw.explained_variance_}")

# ============================================================
# 三、标准化后的 PCA (推荐做法)
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\n" + "=" * 60)
print("【标准化后的 PCA 结果】")
print(f"各主成分方差解释比例: {pca.explained_variance_ratio_}")
print(f"累积方差解释比例: {np.cumsum(pca.explained_variance_ratio_)}")

for i, (var_ratio, cum_ratio) in enumerate(zip(pca.explained_variance_ratio_,
                                                 np.cumsum(pca.explained_variance_ratio_))):
    print(f"  PC{i + 1}: 方差解释比例={var_ratio:.4f}, 累积={cum_ratio:.4f}")

# ============================================================
# 四、主成分载荷分析
# ============================================================

print("\n" + "=" * 60)
print("【主成分载荷 (成分矩阵)】")
print("载荷值反映原始特征对主成分的贡献程度")
print(f"{'特征':<25} {'PC1':>10} {'PC2':>10}")
print("-" * 47)
for i, name in enumerate(iris.feature_names):
    print(f"{name:<25} {pca.components_[0, i]:>10.4f} {pca.components_[1, i]:>10.4f}")

# ============================================================
# 五、指定 n_components 降维
# ============================================================

# 保留 95% 方差
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print("\n" + "=" * 60)
print("【PCA 保留 95% 方差】")
print(f"原始维度: {X_scaled.shape[1]}")
print(f"降维后维度: {X_pca_95.shape[1]}")
print(f"实际保留方差: {pca_95.explained_variance_ratio_.sum():.4f}")

# 指定保留 2 个主成分
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print("\n" + "=" * 60)
print("【PCA 保留 2 个主成分】")
print(f"降维后形状: {X_pca_2d.shape}")
print(f"方差解释比例: {pca_2d.explained_variance_ratio_}")
print(f"累积方差解释比例: {pca_2d.explained_variance_ratio_.sum():.4f}")

# ============================================================
# 六、2D 降维可视化
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 不标准化的 PCA 2D
for cls_idx, cls_name in enumerate(iris.target_names):
    mask = y == cls_idx
    axes[0].scatter(X_pca_raw[mask, 0], X_pca_raw[mask, 1],
                    label=cls_name, s=50, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca_raw.explained_variance_ratio_[0]:.2%})', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca_raw.explained_variance_ratio_[1]:.2%})', fontsize=12)
axes[0].set_title('PCA 2D 可视化 (未标准化)', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 标准化后的 PCA 2D
for cls_idx, cls_name in enumerate(iris.target_names):
    mask = y == cls_idx
    axes[1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                    label=cls_name, s=50, alpha=0.7)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})', fontsize=12)
axes[1].set_title('PCA 2D 可视化 (标准化后)', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_2D可视化.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、方差解释比例可视化
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 不标准化
n_components_raw = len(pca_raw.explained_variance_ratio_)
x_raw = np.arange(1, n_components_raw + 1)
axes[0].bar(x_raw, pca_raw.explained_variance_ratio_, color='steelblue',
            alpha=0.7, label='各主成分方差解释比例')
axes[0].plot(x_raw, np.cumsum(pca_raw.explained_variance_ratio_), 'ro-',
             label='累积方差解释比例')
axes[0].axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%阈值')
axes[0].set_xlabel('主成分', fontsize=12)
axes[0].set_ylabel('方差解释比例', fontsize=12)
axes[0].set_title('方差解释比例 (未标准化)', fontsize=13)
axes[0].set_xticks(x_raw)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 标准化后
n_components = len(pca.explained_variance_ratio_)
x_vals = np.arange(1, n_components + 1)
axes[1].bar(x_vals, pca.explained_variance_ratio_, color='darkorange',
            alpha=0.7, label='各主成分方差解释比例')
axes[1].plot(x_vals, np.cumsum(pca.explained_variance_ratio_), 'ro-',
             label='累积方差解释比例')
axes[1].axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%阈值')
axes[1].set_xlabel('主成分', fontsize=12)
axes[1].set_ylabel('方差解释比例', fontsize=12)
axes[1].set_title('方差解释比例 (标准化后)', fontsize=13)
axes[1].set_xticks(x_vals)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_方差解释比例.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 八、主成分载荷可视化
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

for i, name in enumerate(iris.feature_names):
    ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
             head_width=0.03, head_length=0.02, fc='red', ec='red')
    ax.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15,
            name, fontsize=11, ha='center', color='red')

circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
ax.add_patch(circle)

for cls_idx, cls_name in enumerate(iris.target_names):
    mask = y == cls_idx
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
               label=cls_name, s=30, alpha=0.5)

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})', fontsize=12)
ax.set_title('PCA 双标图: 样本散点 + 特征载荷向量', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('PCA_双标图.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("主成分分析 PCA 降维演示完成!")
