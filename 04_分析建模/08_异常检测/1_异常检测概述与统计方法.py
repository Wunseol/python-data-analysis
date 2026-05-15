# 数据来源: 模拟正态分布数据与异常注入
# 依赖库最低版本要求: scikit-learn>=1.3, numpy>=1.24, scipy>=1.10

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# 异常类型说明
print("=" * 50)
print("异常检测主要类型:")
print("  1. 点异常(Point): 单个数据点偏离正常范围")
print("  2. 上下文异常(Contextual): 在特定上下文中异常")
print("  3. 集合异常(Collective): 单个正常但组合异常")
print("=" * 50)

# 生成正常数据
n_normal = 200
data_normal = np.random.normal(loc=50, scale=5, size=n_normal)
print(f"\n正常数据: 均值={data_normal.mean():.2f}, 标准差={data_normal.std():.2f}")

# 注入异常点
n_outliers = 10
outliers = np.random.uniform(70, 90, size=n_outliers)
data = np.concatenate([data_normal, outliers])
labels = np.array([0] * n_normal + [1] * n_outliers)
print(f"总数据量: {len(data)}, 异常点数: {n_outliers}")

df = pd.DataFrame({"值": data, "是否异常": labels})

# === 3σ原则 ===
print("\n=== 3σ原则 ===")
mean_val = data.mean()
std_val = data.std()
lower_3sigma = mean_val - 3 * std_val
upper_3sigma = mean_val + 3 * std_val
print(f"均值: {mean_val:.2f}, 标准差: {std_val:.2f}")
print(f"3σ范围: [{lower_3sigma:.2f}, {upper_3sigma:.2f}]")

anomaly_3sigma = (data < lower_3sigma) | (data > upper_3sigma)
print(f"3σ检测异常数: {anomaly_3sigma.sum()}")
print(f"3σ检测到的异常值: {data[anomaly_3sigma].round(2)}")

# === Z-score方法 ===
print("\n=== Z-score方法 ===")
z_scores = (data - mean_val) / std_val
threshold = 3
anomaly_zscore = np.abs(z_scores) > threshold
print(f"Z-score阈值: {threshold}")
print(f"Z-score检测异常数: {anomaly_zscore.sum()}")
print(f"Z-score检测到的异常值: {data[anomaly_zscore].round(2)}")

# 不同阈值对比
print("\n--- 不同Z-score阈值对比 ---")
for t in [2, 2.5, 3, 3.5]:
    detected = np.abs(z_scores) > t
    tp = (detected & labels.astype(bool)).sum()
    fp = (detected & ~labels.astype(bool)).sum()
    fn = (~detected & labels.astype(bool)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  阈值={t}: 检出{detected.sum()}个, Precision={precision:.2f}, Recall={recall:.2f}")

# === IQR方法 ===
print("\n=== IQR方法 ===")
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_iqr = q1 - 1.5 * iqr
upper_iqr = q3 + 1.5 * iqr
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"IQR范围: [{lower_iqr:.2f}, {upper_iqr:.2f}]")

anomaly_iqr = (data < lower_iqr) | (data > upper_iqr)
print(f"IQR检测异常数: {anomaly_iqr.sum()}")
print(f"IQR检测到的异常值: {data[anomaly_iqr].round(2)}")

# 三种方法对比
print("\n=== 三种方法对比 ===")
print(f"3σ原则: 检出{anomaly_3sigma.sum()}个异常")
print(f"Z-score: 检出{anomaly_zscore.sum()}个异常")
print(f"IQR方法: 检出{anomaly_iqr.sum()}个异常")
print("注: IQR对非正态分布更鲁棒, 3σ/Z-score假设正态分布")
