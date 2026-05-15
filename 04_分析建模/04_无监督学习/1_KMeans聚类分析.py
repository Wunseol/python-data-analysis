# 数据来源: sklearn.datasets.make_blobs (合成聚类数据)

"""
KMeans 聚类分析
================
本案例演示:
1. KMeans 聚类算法
2. n_clusters 簇数的选择
3. 肘部法则 (inertia_)
4. 轮廓系数 silhouette_score
5. 聚类中心与可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、生成合成数据
# ============================================================

X, y_true = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1.5,
    random_state=42
)

print("=" * 60)
print("【合成数据信息】")
print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"真实簇数: {len(np.unique(y_true))}")
print(f"真实簇分布: {np.bincount(y_true)}")

# ============================================================
# 二、KMeans 聚类
# ============================================================

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

y_pred = kmeans.labels_
centers = kmeans.cluster_centers_

print("\n" + "=" * 60)
print("【KMeans 聚类结果 (K=4)】")
print(f"聚类标签: {np.unique(y_pred)}")
print(f"聚类中心:\n{centers}")
print(f"惯性 (inertia): {kmeans.inertia_:.2f}")
print(f"迭代次数: {kmeans.n_iter_}")

# ============================================================
# 三、肘部法则 — 选择最优 K
# ============================================================

K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, km.labels_))

print("\n" + "=" * 60)
print("【肘部法则与轮廓系数】")
print(f"{'K值':<6} {'惯性(inertia)':>15} {'轮廓系数':>12}")
print("-" * 35)
for k, inertia, sil in zip(K_range, inertias, silhouette_scores):
    print(f"{k:<6} {inertia:>15.2f} {sil:>12.4f}")

best_k_sil = list(K_range)[np.argmax(silhouette_scores)]
print(f"\n轮廓系数最优 K 值: {best_k_sil}, 轮廓系数: {max(silhouette_scores):.4f}")

# ============================================================
# 四、不同 K 值的聚类结果可视化
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
k_values = [2, 3, 4, 5, 6, 7]

for idx, k in enumerate(k_values):
    ax = axes[idx // 3, idx % 3]
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)

    scatter = ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=plt.cm.Set1,
                         s=20, alpha=0.6)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='black', marker='X', s=200, label='聚类中心')
    ax.set_title(f'K={k}, 轮廓系数={sil:.4f}', fontsize=13)
    ax.legend(fontsize=9)

plt.suptitle('KMeans 不同 K 值的聚类结果', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('KMeans_不同K值.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 五、肘部法则与轮廓系数图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'o-', color='steelblue', markersize=8)
axes[0].set_xlabel('K值 (簇数)', fontsize=12)
axes[0].set_ylabel('惯性 (inertia)', fontsize=12)
axes[0].set_title('肘部法则: 惯性随K值变化', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].annotate('肘部', xy=(4, inertias[2]), xytext=(6, inertias[2] + 200),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=12, color='red')

axes[1].plot(K_range, silhouette_scores, 's-', color='darkorange', markersize=8)
axes[1].axvline(x=best_k_sil, color='red', linestyle='--', alpha=0.7,
                label=f'最优K={best_k_sil}')
axes[1].set_xlabel('K值 (簇数)', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数随K值变化', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('KMeans_肘部法则与轮廓系数.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 六、最优聚类结果详细分析
# ============================================================

km_best = KMeans(n_clusters=best_k_sil, random_state=42, n_init=10)
km_best.fit(X)

print("\n" + "=" * 60)
print(f"【最优聚类结果 (K={best_k_sil})】")
for i in range(best_k_sil):
    cluster_size = np.sum(km_best.labels_ == i)
    cluster_center = km_best.cluster_centers_[i]
    print(f"簇 {i}: 样本数={cluster_size}, 中心=({cluster_center[0]:.2f}, {cluster_center[1]:.2f})")

fig, ax = plt.subplots(figsize=(10, 7))

for i in range(best_k_sil):
    mask = km_best.labels_ == i
    ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.6,
               label=f'簇 {i} (n={mask.sum()})')

ax.scatter(km_best.cluster_centers_[:, 0], km_best.cluster_centers_[:, 1],
           c='black', marker='X', s=200, label='聚类中心', zorder=5)
ax.set_xlabel('特征1', fontsize=12)
ax.set_ylabel('特征2', fontsize=12)
ax.set_title(f'KMeans 最优聚类结果 (K={best_k_sil}, 轮廓系数={max(silhouette_scores):.4f})',
             fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('KMeans_最优聚类.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、聚类与真实标签对比
# ============================================================

print("\n" + "=" * 60)
print("【聚类标签 vs 真实标签分布】")
for true_label in np.unique(y_true):
    mask = y_true == true_label
    pred_labels = km_best.labels_[mask]
    print(f"真实簇 {true_label} (n={mask.sum()}): 聚类标签分布 = {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

print("\n" + "=" * 60)
print("KMeans 聚类分析演示完成!")
