# 数据来源: sklearn内置数据集与模拟数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 生成模拟数据（3个簇 + 异常点）
X_cluster, _ = make_blobs(n_samples=[120, 100, 80], centers=[[-3, -3], [3, 3], [7, 0]],
                          cluster_std=0.8, random_state=42)
X_outliers = np.random.uniform(low=-8, high=12, size=(20, 2))
X = np.vstack([X_cluster, X_outliers])
y_true = np.array([1] * 300 + [-1] * 20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"数据量: {len(X)}, 正常: {(y_true == 1).sum()}, 异常: {(y_true == -1).sum()}")

# === KMeans聚类距离异常检测 ===
print("\n=== KMeans聚类距离异常检测 ===")

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_scaled)
distances = kmeans.transform(X_scaled)
min_distances = distances.min(axis=1)

# 基于聚类距离的异常分数
threshold_95 = np.percentile(min_distances, 95)
kmeans_pred = np.where(min_distances > threshold_95, -1, 1)

print(f"聚类数: {n_clusters}")
print(f"距离阈值(95%分位): {threshold_95:.4f}")
print(f"KMeans检测异常数: {(kmeans_pred == -1).sum()}")

tp = ((kmeans_pred == -1) & (y_true == -1)).sum()
fp = ((kmeans_pred == -1) & (y_true == 1)).sum()
fn = ((kmeans_pred == 1) & (y_true == -1)).sum()
p = tp / (tp + fp) if (tp + fp) > 0 else 0
r = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Precision={p:.2f}, Recall={r:.2f}")

# 不同聚类数对比
print("\n--- 不同聚类数对比 ---")
for nc in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=nc, random_state=42, n_init=10)
    km.fit(X_scaled)
    dist = km.transform(X_scaled).min(axis=1)
    thresh = np.percentile(dist, 95)
    pred = np.where(dist > thresh, -1, 1)
    tp_k = ((pred == -1) & (y_true == -1)).sum()
    fp_k = ((pred == -1) & (y_true == 1)).sum()
    fn_k = ((pred == 1) & (y_true == -1)).sum()
    p_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0
    r_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0
    print(f"  K={nc}: 检出{(pred == -1).sum()}, P={p_k:.2f}, R={r_k:.2f}")

# === DBSCAN噪声点检测 ===
print("\n=== DBSCAN噪声点检测 ===")

# eps参数调优
print("--- eps参数调优 ---")
for eps_val in [0.3, 0.5, 0.8, 1.0, 1.5]:
    db = DBSCAN(eps=eps_val, min_samples=5)
    labels_db = db.fit_predict(X_scaled)
    n_noise = (labels_db == -1).sum()
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    db_pred = np.where(labels_db == -1, -1, 1)
    tp_d = ((db_pred == -1) & (y_true == -1)).sum()
    fp_d = ((db_pred == -1) & (y_true == 1)).sum()
    fn_d = ((db_pred == 1) & (y_true == -1)).sum()
    p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
    r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
    print(f"  eps={eps_val}: 簇={n_clusters_db}, 噪声={n_noise}, P={p_d:.2f}, R={r_d:.2f}")

# min_samples参数调优
print("\n--- min_samples参数调优 ---")
for ms in [3, 5, 7, 10, 15]:
    db = DBSCAN(eps=0.8, min_samples=ms)
    labels_db = db.fit_predict(X_scaled)
    n_noise = (labels_db == -1).sum()
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    db_pred = np.where(labels_db == -1, -1, 1)
    tp_d = ((db_pred == -1) & (y_true == -1)).sum()
    fp_d = ((db_pred == -1) & (y_true == 1)).sum()
    fn_d = ((db_pred == 1) & (y_true == -1)).sum()
    p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
    r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
    print(f"  min_samples={ms}: 簇={n_clusters_db}, 噪声={n_noise}, P={p_d:.2f}, R={r_d:.2f}")

# === 聚类半径方法 ===
print("\n=== 聚类半径方法 ===")
kmeans_r = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_r.fit(X_scaled)
cluster_labels = kmeans_r.labels_
centroids = kmeans_r.cluster_centers_

# 计算每个簇的半径(到中心距离的95%分位数)
cluster_radii = {}
for c in range(3):
    mask = cluster_labels == c
    dists = np.linalg.norm(X_scaled[mask] - centroids[c], axis=1)
    cluster_radii[c] = np.percentile(dists, 95)
    print(f"  簇{c}: 样本数={mask.sum()}, 半径(95%)={cluster_radii[c]:.4f}")

# 超出簇半径的样本为异常
radius_pred = np.ones(len(X_scaled), dtype=int)
for i in range(len(X_scaled)):
    c = cluster_labels[i]
    dist = np.linalg.norm(X_scaled[i] - centroids[c])
    if dist > cluster_radii[c]:
        radius_pred[i] = -1

tp_r = ((radius_pred == -1) & (y_true == -1)).sum()
fp_r = ((radius_pred == -1) & (y_true == 1)).sum()
fn_r = ((radius_pred == 1) & (y_true == -1)).sum()
p_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0
r_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
print(f"\n聚类半径方法: 检出{(radius_pred == -1).sum()}, P={p_r:.2f}, R={r_r:.2f}")

# 三种方法对比
print("\n=== 三种聚类异常检测方法对比 ===")
print("KMeans距离: 基于到最近中心的距离, 适合球形簇")
print("DBSCAN噪声: 基于密度连通性, 适合任意形状簇")
print("聚类半径: 基于簇内距离分布, 更灵活的阈值")
