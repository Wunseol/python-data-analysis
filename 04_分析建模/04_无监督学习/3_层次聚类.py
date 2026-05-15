# 依赖库最低版本要求: scikit-learn>=1.3, numpy>=1.24, matplotlib>=3.7, scipy>=1.10
# 数据来源: sklearn.datasets.make_blobs (合成聚类数据), sklearn.datasets.load_iris (鸢尾花数据集)

"""
层次聚类
========
本案例演示:
1. AgglomerativeClustering 凝聚层次聚类
2. n_clusters 簇数参数
3. linkage 连接方式 (ward, complete, average, single)
4. scipy.cluster.hierarchy 树状图 (dendrogram) 可视化
5. 不同连接方式的对比
6. distance_threshold 距离阈值
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、生成合成数据
# ============================================================

X_blobs, y_blobs = make_blobs(
    n_samples=300,
    n_features=2,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

print("=" * 60)
print("【合成数据信息】")
print(f"样本数: {X_blobs.shape[0]}")
print(f"特征数: {X_blobs.shape[1]}")
print(f"真实簇数: {len(np.unique(y_blobs))}")

# ============================================================
# 二、AgglomerativeClustering 基本用法
# ============================================================

agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_pred = agg.fit_predict(X_blobs)

print("\n" + "=" * 60)
print("【AgglomerativeClustering 聚类结果 (n_clusters=4, linkage=ward)】")
print(f"聚类标签: {np.unique(y_pred)}")
print(f"各簇样本数: {np.bincount(y_pred)}")
sil = silhouette_score(X_blobs, y_pred)
print(f"轮廓系数: {sil:.4f}")
print(f"连接方式: {agg.linkage}")
print(f"子节点数: {agg.n_connected_components_}")

# ============================================================
# 三、树状图 (Dendrogram) 可视化
# ============================================================

Z = linkage(X_blobs, method='ward')

fig, ax = plt.subplots(figsize=(14, 7))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=90,
           leaf_font_size=9, show_contracted=True)
ax.set_xlabel('样本索引 (或簇大小)', fontsize=12)
ax.set_ylabel('距离', fontsize=12)
ax.set_title('层次聚类树状图 (Ward连接)', fontsize=14)
ax.axhline(y=Z[-4, 2], color='red', linestyle='--', alpha=0.7,
           label=f'切割线 (距离={Z[-4, 2]:.1f}, 4簇)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_树状图.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 四、不同连接方式 (linkage) 对比
# ============================================================

linkage_methods = ['ward', 'complete', 'average', 'single']

print("\n" + "=" * 60)
print("【不同连接方式对比 (n_clusters=4)】")
print(f"{'连接方式':<12} {'轮廓系数':>10} {'各簇样本数':>30}")
print("-" * 55)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, method in enumerate(linkage_methods):
    agg_m = AgglomerativeClustering(n_clusters=4, linkage=method)
    y_m = agg_m.fit_predict(X_blobs)
    sil_m = silhouette_score(X_blobs, y_m)
    counts = np.bincount(y_m)

    print(f"{method:<12} {sil_m:>10.4f} {str(counts):>30}")

    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_m,
                         cmap=plt.cm.Set1, s=25, alpha=0.7)
    ax.set_title(f'linkage={method}, 轮廓系数={sil_m:.4f}', fontsize=13)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.grid(True, alpha=0.3)

plt.suptitle('层次聚类: 不同连接方式对比', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_连接方式对比.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 五、不同簇数的层次聚类
# ============================================================

n_clusters_range = range(2, 8)
sil_scores = []

for n_c in n_clusters_range:
    agg_nc = AgglomerativeClustering(n_clusters=n_c, linkage='ward')
    y_nc = agg_nc.fit_predict(X_blobs)
    sil_scores.append(silhouette_score(X_blobs, y_nc))

best_n = list(n_clusters_range)[np.argmax(sil_scores)]

print("\n" + "=" * 60)
print("【不同簇数的轮廓系数 (linkage=ward)】")
print(f"{'簇数':<6} {'轮廓系数':>10}")
print("-" * 18)
for n_c, sil_sc in zip(n_clusters_range, sil_scores):
    print(f"{n_c:<6} {sil_sc:>10.4f}")
print(f"\n最优簇数: {best_n}, 轮廓系数: {max(sil_scores):.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n_clusters_range, sil_scores, 'o-', color='steelblue', markersize=8)
ax.axvline(x=best_n, color='red', linestyle='--', alpha=0.7,
           label=f'最优n_clusters={best_n}')
ax.set_xlabel('簇数', fontsize=12)
ax.set_ylabel('轮廓系数', fontsize=12)
ax.set_title('层次聚类: 轮廓系数随簇数变化', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(list(n_clusters_range))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_轮廓系数.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 六、distance_threshold 距离阈值
# ============================================================

agg_dt = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=30,
    linkage='ward'
)
y_dt = agg_dt.fit_predict(X_blobs)

print("\n" + "=" * 60)
print("【distance_threshold 模式 (threshold=30, linkage=ward)】")
print(f"自动确定簇数: {agg_dt.n_clusters_}")
print(f"各簇样本数: {np.bincount(y_dt)}")
sil_dt = silhouette_score(X_blobs, y_dt)
print(f"轮廓系数: {sil_dt:.4f}")

thresholds = [15, 25, 35, 50]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, thr in enumerate(thresholds):
    agg_t = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=thr,
        linkage='ward'
    )
    y_t = agg_t.fit_predict(X_blobs)
    n_found = agg_t.n_clusters_

    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_t,
                         cmap=plt.cm.Set1, s=25, alpha=0.7)
    ax.set_title(f'threshold={thr}, 自动簇数={n_found}', fontsize=13)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.grid(True, alpha=0.3)

plt.suptitle('层次聚类: 不同距离阈值的效果', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_距离阈值.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、鸢尾花数据集层次聚类
# ============================================================

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

Z_iris = linkage(X_iris_scaled, method='ward')

fig, ax = plt.subplots(figsize=(14, 7))
dendrogram(Z_iris, ax=ax, truncate_mode='lastp', p=20, leaf_rotation=90,
           leaf_font_size=9, show_contracted=True)
ax.set_xlabel('样本索引 (或簇大小)', fontsize=12)
ax.set_ylabel('距离', fontsize=12)
ax.set_title('鸢尾花数据集层次聚类树状图', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_鸢尾花树状图.png', dpi=150, bbox_inches='tight')
plt.show()

agg_iris = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_iris_pred = agg_iris.fit_predict(X_iris_scaled)

print("\n" + "=" * 60)
print("【鸢尾花数据集层次聚类结果 (n_clusters=3, linkage=ward)】")
print(f"聚类标签: {np.unique(y_iris_pred)}")
print(f"各簇样本数: {np.bincount(y_iris_pred)}")
sil_iris = silhouette_score(X_iris_scaled, y_iris_pred)
print(f"轮廓系数: {sil_iris:.4f}")

print("\n聚类标签 vs 真实标签分布:")
for true_label in np.unique(y_iris):
    mask = y_iris == true_label
    pred_labels = y_iris_pred[mask]
    print(f"  真实类别 {iris.target_names[true_label]} (n={mask.sum()}): "
          f"聚类标签分布 = {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

pca_vis = PCA(n_components=2)
X_iris_pca = pca_vis.fit_transform(X_iris_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for cls_idx, cls_name in enumerate(iris.target_names):
    mask = y_iris == cls_idx
    axes[0].scatter(X_iris_pca[mask, 0], X_iris_pca[mask, 1],
                    label=cls_name, s=40, alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]:.2%})', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]:.2%})', fontsize=12)
axes[0].set_title('真实标签', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

for cluster_id in np.unique(y_iris_pred):
    mask = y_iris_pred == cluster_id
    axes[1].scatter(X_iris_pca[mask, 0], X_iris_pca[mask, 1],
                    label=f'簇 {cluster_id} (n={mask.sum()})', s=40, alpha=0.7)
axes[1].set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]:.2%})', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]:.2%})', fontsize=12)
axes[1].set_title(f'层次聚类 (轮廓系数={sil_iris:.4f})', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_鸢尾花结果.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 八、不同连接方式在鸢尾花数据集上的对比
# ============================================================

from sklearn.decomposition import PCA

pca_iris = PCA(n_components=2)
X_iris_2d = pca_iris.fit_transform(X_iris_scaled)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

print("\n" + "=" * 60)
print("【鸢尾花数据集: 不同连接方式对比 (n_clusters=3)】")
print(f"{'连接方式':<12} {'轮廓系数':>10} {'各簇样本数':>30}")
print("-" * 55)

for idx, method in enumerate(linkage_methods):
    agg_iris_m = AgglomerativeClustering(n_clusters=3, linkage=method)
    y_iris_m = agg_iris_m.fit_predict(X_iris_scaled)
    sil_iris_m = silhouette_score(X_iris_scaled, y_iris_m)
    counts = np.bincount(y_iris_m)

    print(f"{method:<12} {sil_iris_m:>10.4f} {str(counts):>30}")

    ax = axes[idx // 2, idx % 2]
    for cluster_id in np.unique(y_iris_m):
        mask = y_iris_m == cluster_id
        ax.scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1],
                   label=f'簇 {cluster_id} (n={mask.sum()})', s=30, alpha=0.7)
    ax.set_title(f'linkage={method}, 轮廓系数={sil_iris_m:.4f}', fontsize=13)
    ax.set_xlabel(f'PC1 ({pca_iris.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_iris.explained_variance_ratio_[1]:.2%})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('鸢尾花: 不同连接方式对比', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '层次聚类_鸢尾花连接方式对比.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("层次聚类演示完成!")
