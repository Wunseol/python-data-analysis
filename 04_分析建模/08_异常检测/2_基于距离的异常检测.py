# 数据来源: sklearn内置数据集与模拟数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 生成模拟数据
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
X_outliers = np.random.uniform(low=-8, high=8, size=(20, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1] * 300 + [-1] * 20)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"数据量: {len(X)}, 正常: {(y_true == 1).sum()}, 异常: {(y_true == -1).sum()}")

# === KNN距离异常检测 ===
print("\n=== KNN距离异常检测 ===")

k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)

# KNN异常分数: 到第k个近邻的距离
knn_scores = distances[:, -1]
knn_threshold = np.percentile(knn_scores, 95)
knn_pred = np.where(knn_scores > knn_threshold, -1, 1)

print(f"K={k}, 阈值(95%分位数): {knn_threshold:.4f}")
print(f"KNN检测异常数: {(knn_pred == -1).sum()}")

# 平均KNN距离
avg_knn_scores = distances.mean(axis=1)
avg_threshold = np.percentile(avg_knn_scores, 95)
avg_pred = np.where(avg_knn_scores > avg_threshold, -1, 1)
print(f"平均KNN距离检测异常数: {(avg_pred == -1).sum()}")

# 不同K值对比
print("\n--- 不同K值对比 ---")
for k_val in [3, 5, 10, 20]:
    nn_k = NearestNeighbors(n_neighbors=k_val)
    nn_k.fit(X_scaled)
    dist_k, _ = nn_k.kneighbors(X_scaled)
    scores_k = dist_k[:, -1]
    thresh_k = np.percentile(scores_k, 95)
    pred_k = np.where(scores_k > thresh_k, -1, 1)
    tp = ((pred_k == -1) & (y_true == -1)).sum()
    fp = ((pred_k == -1) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (y_true == -1).sum()
    print(f"  K={k_val}: 检出{(pred_k == -1).sum()}, Precision={precision:.2f}, Recall={recall:.2f}")

# === 局部离群因子LOF ===
print("\n=== 局部离群因子LOF ===")

for n_val in [5, 10, 20, 35]:
    lof = LocalOutlierFactor(n_neighbors=n_val, contamination=0.06)
    lof_pred = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
    tp = ((lof_pred == -1) & (y_true == -1)).sum()
    fp = ((lof_pred == -1) & (y_true == 1)).sum()
    fn = ((lof_pred == 1) & (y_true == -1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  n_neighbors={n_val}: 检出{(lof_pred == -1).sum()}, "
          f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

# LOF异常分数分析
lof_final = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
lof_final.fit(X_scaled)
lof_scores_final = -lof_final.negative_outlier_factor_
print(f"\nLOF分数统计:")
print(f"  正常样本LOF均值: {lof_scores_final[y_true == 1].mean():.4f}")
print(f"  异常样本LOF均值: {lof_scores_final[y_true == -1].mean():.4f}")
print(f"  LOF≈1表示密度与邻居相近, LOF>>1表示密度远低于邻居")

# KNN vs LOF对比
print("\n=== KNN vs LOF 对比 ===")
print("KNN: 基于全局距离, 对全局异常点敏感")
print("LOF: 基于局部密度, 对局部异常点更敏感")
print("LOF优势: 能处理不同密度区域中的异常点")
