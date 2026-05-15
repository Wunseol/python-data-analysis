# 数据来源: sklearn.datasets.make_moons (月牙形数据), sklearn.datasets.make_circles (环形数据)

"""
DBSCAN 密度聚类
===============
本案例演示:
1. DBSCAN 算法原理与基本用法
2. eps 和 min_samples 参数
3. 核心点、边界点、噪声点
4. 与 KMeans 的对比
5. eps 参数对聚类效果的影响
6. Silhouette 轮廓系数评估
7. 处理非凸形状数据
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、生成非凸形状数据
# ============================================================

X_moons, y_moons = make_moons(n_samples=500, noise=0.08, random_state=42)
X_circles, y_circles = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)

print("=" * 60)
print("【数据集信息】")
print(f"月牙形数据: 样本数={X_moons.shape[0]}, 真实簇数=2")
print(f"环形数据:   样本数={X_circles.shape[0]}, 真实簇数=2")

# ============================================================
# 二、DBSCAN 基本用法
# ============================================================

dbscan = DBSCAN(eps=0.2, min_samples=5)
y_moons_pred = dbscan.fit_predict(X_moons)

n_clusters = len(set(y_moons_pred)) - (1 if -1 in y_moons_pred else 0)
n_noise = np.sum(y_moons_pred == -1)

print("\n" + "=" * 60)
print("【DBSCAN 基本聚类结果 (月牙形数据, eps=0.2, min_samples=5)】")
print(f"聚类簇数: {n_clusters}")
print(f"噪声点数: {n_noise} ({n_noise / len(y_moons_pred) * 100:.1f}%)")
print(f"各簇样本数: {dict(zip(*np.unique(y_moons_pred, return_counts=True)))}")

core_mask = np.zeros(len(y_moons_pred), dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
border_mask = (y_moons_pred != -1) & ~core_mask
noise_mask = y_moons_pred == -1

print(f"核心点数: {core_mask.sum()}")
print(f"边界点数: {border_mask.sum()}")
print(f"噪声点数: {noise_mask.sum()}")

# ============================================================
# 三、核心点、边界点、噪声点可视化
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(X_moons[core_mask, 0], X_moons[core_mask, 1],
           c=y_moons_pred[core_mask], cmap=plt.cm.Set1, s=30, alpha=0.7,
           label=f'核心点 (n={core_mask.sum()})', edgecolors='k', linewidths=0.3)
ax.scatter(X_moons[border_mask, 0], X_moons[border_mask, 1],
           c=y_moons_pred[border_mask], cmap=plt.cm.Set1, s=30, alpha=0.7,
           marker='s', label=f'边界点 (n={border_mask.sum()})', edgecolors='k', linewidths=0.3)
ax.scatter(X_moons[noise_mask, 0], X_moons[noise_mask, 1],
           c='gray', s=50, alpha=0.8, marker='x',
           label=f'噪声点 (n={noise_mask.sum()})')

ax.set_xlabel('特征1', fontsize=12)
ax.set_ylabel('特征2', fontsize=12)
ax.set_title('DBSCAN: 核心点、边界点与噪声点', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'DBSCAN_核心边界噪声点.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 四、DBSCAN vs KMeans 对比
# ============================================================

kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans_moons = kmeans_moons.fit_predict(X_moons)

kmeans_circles = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans_circles = kmeans_circles.fit_predict(X_circles)

dbscan_circles = DBSCAN(eps=0.15, min_samples=5)
y_dbscan_circles = dbscan_circles.fit_predict(X_circles)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans_moons,
                    cmap=plt.cm.Set1, s=25, alpha=0.7)
axes[0, 0].set_title('KMeans - 月牙形数据', fontsize=13)
axes[0, 0].set_xlabel('特征1')
axes[0, 0].set_ylabel('特征2')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons_pred,
                    cmap=plt.cm.Set1, s=25, alpha=0.7)
n_c_moons = len(set(y_moons_pred)) - (1 if -1 in y_moons_pred else 0)
axes[0, 1].set_title(f'DBSCAN - 月牙形数据 (簇数={n_c_moons})', fontsize=13)
axes[0, 1].set_xlabel('特征1')
axes[0, 1].set_ylabel('特征2')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_kmeans_circles,
                    cmap=plt.cm.Set1, s=25, alpha=0.7)
axes[1, 0].set_title('KMeans - 环形数据', fontsize=13)
axes[1, 0].set_xlabel('特征1')
axes[1, 0].set_ylabel('特征2')
axes[1, 0].grid(True, alpha=0.3)

n_c_circles = len(set(y_dbscan_circles)) - (1 if -1 in y_dbscan_circles else 0)
axes[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_dbscan_circles,
                    cmap=plt.cm.Set1, s=25, alpha=0.7)
axes[1, 1].set_title(f'DBSCAN - 环形数据 (簇数={n_c_circles})', fontsize=13)
axes[1, 1].set_xlabel('特征1')
axes[1, 1].set_ylabel('特征2')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('DBSCAN vs KMeans: 非凸形状数据处理能力对比', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'DBSCAN_vs_KMeans.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("【DBSCAN vs KMeans 对比】")
print("KMeans 无法正确处理非凸形状 (月牙、环形), 只能按距离划分")
print("DBSCAN 基于密度, 能够识别任意形状的簇")

# ============================================================
# 五、eps 参数对聚类效果的影响
# ============================================================

eps_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]

print("\n" + "=" * 60)
print("【eps 参数对聚类效果的影响 (月牙形数据, min_samples=5)】")
print(f"{'eps':<8} {'簇数':>6} {'噪声点数':>10} {'噪声比例':>10}")
print("-" * 38)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, eps_val in enumerate(eps_values):
    db = DBSCAN(eps=eps_val, min_samples=5)
    y_db = db.fit_predict(X_moons)
    n_c = len(set(y_db)) - (1 if -1 in y_db else 0)
    n_n = np.sum(y_db == -1)
    noise_ratio = n_n / len(y_db) * 100

    print(f"{eps_val:<8.2f} {n_c:>6} {n_n:>10} {noise_ratio:>9.1f}%")

    ax = axes[idx // 3, idx % 3]
    scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_db,
                         cmap=plt.cm.Set1, s=20, alpha=0.7)
    ax.set_title(f'eps={eps_val}, 簇数={n_c}, 噪声={n_n}', fontsize=12)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.grid(True, alpha=0.3)

plt.suptitle('DBSCAN: 不同 eps 值的聚类效果', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'DBSCAN_eps影响.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 六、min_samples 参数对聚类效果的影响
# ============================================================

min_samples_values = [3, 5, 10, 15, 20, 30]

print("\n" + "=" * 60)
print("【min_samples 参数对聚类效果的影响 (月牙形数据, eps=0.2)】")
print(f"{'min_samples':<14} {'簇数':>6} {'噪声点数':>10} {'噪声比例':>10}")
print("-" * 44)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, ms in enumerate(min_samples_values):
    db = DBSCAN(eps=0.2, min_samples=ms)
    y_db = db.fit_predict(X_moons)
    n_c = len(set(y_db)) - (1 if -1 in y_db else 0)
    n_n = np.sum(y_db == -1)
    noise_ratio = n_n / len(y_db) * 100

    print(f"{ms:<14} {n_c:>6} {n_n:>10} {noise_ratio:>9.1f}%")

    ax = axes[idx // 3, idx % 3]
    scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=y_db,
                         cmap=plt.cm.Set1, s=20, alpha=0.7)
    ax.set_title(f'min_samples={ms}, 簇数={n_c}, 噪声={n_n}', fontsize=12)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.grid(True, alpha=0.3)

plt.suptitle('DBSCAN: 不同 min_samples 值的聚类效果', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'DBSCAN_min_samples影响.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、轮廓系数评估与参数搜索
# ============================================================

eps_range = np.arange(0.05, 0.55, 0.02)
best_sil = -1
best_eps = 0
best_ms = 5

print("\n" + "=" * 60)
print("【轮廓系数评估: 参数搜索 (月牙形数据)】")
print(f"{'eps':<8} {'min_samples':<14} {'簇数':>6} {'轮廓系数':>10}")
print("-" * 42)

for eps_val in [0.1, 0.15, 0.2, 0.25, 0.3]:
    for ms in [3, 5, 10]:
        db = DBSCAN(eps=eps_val, min_samples=ms)
        y_db = db.fit_predict(X_moons)
        n_c = len(set(y_db)) - (1 if -1 in y_db else 0)
        if n_c >= 2:
            non_noise = y_db != -1
            if non_noise.sum() > 0 and len(set(y_db[non_noise])) >= 2:
                sil = silhouette_score(X_moons[non_noise], y_db[non_noise])
                print(f"{eps_val:<8.2f} {ms:<14} {n_c:>6} {sil:>10.4f}")
                if sil > best_sil:
                    best_sil = sil
                    best_eps = eps_val
                    best_ms = ms

print(f"\n最优参数: eps={best_eps}, min_samples={best_ms}, 轮廓系数={best_sil:.4f}")

# ============================================================
# 八、环形数据集 DBSCAN 聚类
# ============================================================

dbscan_circles_best = DBSCAN(eps=0.15, min_samples=5)
y_circles_pred = dbscan_circles_best.fit_predict(X_circles)

n_c_c = len(set(y_circles_pred)) - (1 if -1 in y_circles_pred else 0)
n_noise_c = np.sum(y_circles_pred == -1)

print("\n" + "=" * 60)
print("【环形数据 DBSCAN 聚类结果 (eps=0.15, min_samples=5)】")
print(f"聚类簇数: {n_c_c}")
print(f"噪声点数: {n_noise_c}")
print(f"各簇样本数: {dict(zip(*np.unique(y_circles_pred, return_counts=True)))}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles,
                cmap=plt.cm.Set1, s=25, alpha=0.7)
axes[0].set_title('环形数据 - 真实标签', fontsize=13)
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles_pred,
                cmap=plt.cm.Set1, s=25, alpha=0.7)
axes[1].set_title(f'DBSCAN 聚类结果 (簇数={n_c_c}, 噪声={n_noise_c})', fontsize=13)
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].grid(True, alpha=0.3)

plt.suptitle('DBSCAN 处理环形数据', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'DBSCAN_环形数据.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 九、含噪声的 blobs 数据集
# ============================================================

X_blobs, y_blobs = make_blobs(n_samples=400, centers=3, cluster_std=1.0,
                                random_state=42)
rng = np.random.RandomState(42)
X_noise = rng.uniform(X_blobs.min(axis=0) - 3, X_blobs.max(axis=0) + 3,
                       size=(30, 2))
X_combined = np.vstack([X_blobs, X_noise])

dbscan_blobs = DBSCAN(eps=1.0, min_samples=5)
y_blobs_pred = dbscan_blobs.fit_predict(X_combined)

n_c_b = len(set(y_blobs_pred)) - (1 if -1 in y_blobs_pred else 0)
n_noise_b = np.sum(y_blobs_pred == -1)

print("\n" + "=" * 60)
print("【含噪声 blobs 数据 DBSCAN 聚类 (eps=1.0, min_samples=5)】")
print(f"聚类簇数: {n_c_b}")
print(f"噪声点数: {n_noise_b} (含人工添加的30个噪声点)")

fig, ax = plt.subplots(figsize=(10, 7))

core_mask_b = np.zeros(len(y_blobs_pred), dtype=bool)
core_mask_b[dbscan_blobs.core_sample_indices_] = True
noise_mask_b = y_blobs_pred == -1
normal_mask_b = (y_blobs_pred != -1) & ~core_mask_b

ax.scatter(X_combined[core_mask_b, 0], X_combined[core_mask_b, 1],
           c=y_blobs_pred[core_mask_b], cmap=plt.cm.Set1, s=25, alpha=0.7,
           label=f'核心点 (n={core_mask_b.sum()})')
ax.scatter(X_combined[normal_mask_b, 0], X_combined[normal_mask_b, 1],
           c=y_blobs_pred[normal_mask_b], cmap=plt.cm.Set1, s=25, alpha=0.7,
           marker='s', label=f'边界点 (n={normal_mask_b.sum()})')
ax.scatter(X_combined[noise_mask_b, 0], X_combined[noise_mask_b, 1],
           c='gray', s=50, alpha=0.8, marker='x',
           label=f'噪声点 (n={noise_mask_b.sum()})')

ax.set_xlabel('特征1', fontsize=12)
ax.set_ylabel('特征2', fontsize=12)
ax.set_title('DBSCAN: 含噪声的 blobs 数据聚类', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'DBSCAN_含噪声blobs.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("DBSCAN 密度聚类演示完成!")
