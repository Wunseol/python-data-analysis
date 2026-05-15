# 数据来源: sklearn内置数据集与模拟数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 生成模拟数据
X_normal, _ = make_blobs(n_samples=400, centers=[[0, 0], [5, 5]], cluster_std=0.8, random_state=42)
X_outliers = np.random.uniform(low=-4, high=10, size=(25, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.array([1] * 400 + [-1] * 25)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"数据量: {len(X)}, 正常: {(y_true == 1).sum()}, 异常: {(y_true == -1).sum()}")

# === Isolation Forest原理 ===
print("\n=== Isolation Forest原理 ===")
print("核心思想: 异常点'少而不同',更容易被孤立")
print("1. 随机选择特征和分割点构建二叉树")
print("2. 异常点路径短(靠近根节点), 正常点路径长")
print("3. 异常分数 = 2^(-E(h(x))/c(n))")
print("   E(h(x)): 样本x的平均路径长度")
print("   c(n): 归一化因子")
print("   分数接近1→异常, 接近0.5→正常")

# === IsolationForest参数调优 ===
print("\n=== IsolationForest参数调优 ===")

# n_estimators调优
print("\n--- n_estimators(树数量) ---")
for n_est in [50, 100, 200, 300]:
    iso = IsolationForest(n_estimators=n_est, contamination=0.06, random_state=42)
    pred = iso.fit_predict(X_scaled)
    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true == 1)).sum()
    fn = ((pred == 1) & (y_true == -1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  n_estimators={n_est}: 检出{(pred == -1).sum()}, P={p:.2f}, R={r:.2f}")

# contamination调优
print("\n--- contamination(污染比例) ---")
for cont in [0.03, 0.05, 0.06, 0.1, 0.15]:
    iso = IsolationForest(n_estimators=100, contamination=cont, random_state=42)
    pred = iso.fit_predict(X_scaled)
    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true == 1)).sum()
    fn = ((pred == 1) & (y_true == -1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  contamination={cont}: 检出{(pred == -1).sum()}, P={p:.2f}, R={r:.2f}, F1={f1:.2f}")

# max_samples调优
print("\n--- max_samples(采样数) ---")
for ms in [64, 128, 256, 'auto']:
    iso = IsolationForest(n_estimators=100, max_samples=ms, contamination=0.06, random_state=42)
    pred = iso.fit_predict(X_scaled)
    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true == 1)).sum()
    fn = ((pred == 1) & (y_true == -1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  max_samples={ms}: 检出{(pred == -1).sum()}, P={p:.2f}, R={r:.2f}")

# === 异常分数解释 ===
print("\n=== 异常分数解释 ===")
iso_final = IsolationForest(n_estimators=200, contamination=0.06, random_state=42)
iso_final.fit(X_scaled)
scores = iso_final.decision_function(X_scaled)
pred_final = iso_final.predict(X_scaled)

print(f"异常分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"正常样本分数均值: {scores[y_true == 1].mean():.4f}")
print(f"异常样本分数均值: {scores[y_true == -1].mean():.4f}")
print(f"分数>0: 倾向正常, 分数<0: 倾向异常")

# 分数分布
print(f"\n--- 异常分数分位数 ---")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {q}%分位数: {np.percentile(scores, q):.4f}")

# === 可视化(文本) ===
print("\n=== 异常检测结果 ===")
anomaly_df = pd.DataFrame({
    "x1": X[:, 0], "x2": X[:, 1],
    "score": scores, "pred": pred_final, "true": y_true
})
print(f"预测为异常的样本:")
anomalous = anomaly_df[anomaly_df["pred"] == -1].sort_values("score")
print(anomalous[["x1", "x2", "score", "true"]].head(10).to_string(index=False, float_format="%.2f"))

tp = ((pred_final == -1) & (y_true == -1)).sum()
fp = ((pred_final == -1) & (y_true == 1)).sum()
fn = ((pred_final == 1) & (y_true == -1)).sum()
print(f"\n最终结果: TP={tp}, FP={fp}, FN={fn}")
