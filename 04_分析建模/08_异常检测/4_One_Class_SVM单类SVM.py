# 数据来源: sklearn内置数据集与模拟数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
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

# === OCSVM原理 ===
print("\n=== One-Class SVM原理 ===")
print("核心思想: 在特征空间中找到一个包含大部分数据的超平面")
print("1. 将数据映射到高维空间")
print("2. 找到与原点最大间隔的超平面")
print("3. 超平面内的数据为正常, 外部为异常")
print("关键参数: nu(异常比例上界), kernel(核函数), gamma(RBF核宽度)")

# === RBF核参数选择 ===
print("\n=== RBF核参数选择 ===")

# nu参数调优
print("\n--- nu参数(异常比例上界) ---")
for nu_val in [0.01, 0.05, 0.1, 0.15, 0.2]:
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu_val)
    pred = ocsvm.fit_predict(X_scaled)
    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true == 1)).sum()
    fn = ((pred == 1) & (y_true == -1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  nu={nu_val}: 检出{(pred == -1).sum()}, P={p:.2f}, R={r:.2f}, F1={f1:.2f}")

# gamma参数调优
print("\n--- gamma参数(RBF核宽度) ---")
for gamma_val in [0.01, 0.05, 0.1, 0.5, 1.0, 'scale', 'auto']:
    ocsvm = OneClassSVM(kernel='rbf', gamma=gamma_val, nu=0.1)
    pred = ocsvm.fit_predict(X_scaled)
    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true == 1)).sum()
    fn = ((pred == 1) & (y_true == -1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  gamma={gamma_val}: 检出{(pred == -1).sum()}, P={p:.2f}, R={r:.2f}")

# === 异常边界分析 ===
print("\n=== 异常边界分析 ===")
ocsvm_final = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
ocsvm_final.fit(X_scaled)
scores = ocsvm_final.decision_function(X_scaled)
pred_final = ocsvm_final.predict(X_scaled)

print(f"决策函数分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"正常样本分数均值: {scores[y_true == 1].mean():.4f}")
print(f"异常样本分数均值: {scores[y_true == -1].mean():.4f}")
print(f"分数>0: 正常区域, 分数<0: 异常区域")

# 边界附近的样本
boundary_samples = np.argsort(np.abs(scores))[:10]
print(f"\n--- 边界附近样本(最不确定) ---")
for idx in boundary_samples:
    print(f"  样本{idx}: 分数={scores[idx]:.4f}, 预测={'异常' if pred_final[idx] == -1 else '正常'}, "
          f"真实={'异常' if y_true[idx] == -1 else '正常'}")

# === 不同核函数对比 ===
print("\n=== 不同核函数对比 ===")
for kernel in ['rbf', 'poly', 'sigmoid', 'linear']:
    try:
        ocsvm_k = OneClassSVM(kernel=kernel, nu=0.1, gamma='scale')
        pred_k = ocsvm_k.fit_predict(X_scaled)
        tp = ((pred_k == -1) & (y_true == -1)).sum()
        fp = ((pred_k == -1) & (y_true == 1)).sum()
        fn = ((pred_k == 1) & (y_true == -1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {kernel}: 检出{(pred_k == -1).sum()}, P={p:.2f}, R={r:.2f}, F1={f1:.2f}")
    except Exception as e:
        print(f"  {kernel}: 失败 - {e}")

# 最终结果
tp = ((pred_final == -1) & (y_true == -1)).sum()
fp = ((pred_final == -1) & (y_true == 1)).sum()
fn = ((pred_final == 1) & (y_true == -1)).sum()
print(f"\n最终OCSVM结果: TP={tp}, FP={fp}, FN={fn}")
