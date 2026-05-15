# 数据来源: 模拟信用卡交易数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

np.random.seed(42)

# === 模拟信用卡交易数据 ===
print("=== 模拟信用卡交易数据 ===")
n_normal = 2000
n_fraud = 40

# 正常交易特征
normal_data = {
    "交易金额": np.random.lognormal(mean=3.5, sigma=1.0, size=n_normal),
    "交易时间小时": np.random.normal(14, 4, n_normal).clip(0, 23),
    "交易频率_24h": np.random.poisson(3, n_normal),
    "距上次交易_分钟": np.random.exponential(120, n_normal),
    "异地交易标志": np.random.choice([0, 1], n_normal, p=[0.95, 0.05]),
    "商户类别数_7d": np.random.poisson(5, n_normal),
    "平均交易金额_7d": np.random.lognormal(mean=3.5, sigma=0.8, size=n_normal),
    "大额交易比例_30d": np.random.beta(2, 10, n_normal),
}

# 欺诈交易特征（分布偏移）
fraud_data = {
    "交易金额": np.random.lognormal(mean=6.0, sigma=1.5, size=n_fraud),
    "交易时间小时": np.random.normal(2, 3, n_fraud).clip(0, 23),
    "交易频率_24h": np.random.poisson(10, n_fraud),
    "距上次交易_分钟": np.random.exponential(5, n_fraud),
    "异地交易标志": np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),
    "商户类别数_7d": np.random.poisson(12, n_fraud),
    "平均交易金额_7d": np.random.lognormal(mean=5.0, sigma=1.2, size=n_fraud),
    "大额交易比例_30d": np.random.beta(8, 2, n_fraud),
}

df_normal = pd.DataFrame(normal_data)
df_normal["标签"] = 0
df_fraud = pd.DataFrame(fraud_data)
df_fraud["标签"] = 1

df = pd.concat([df_normal, df_fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"总交易数: {len(df)}")
print(f"正常交易: {(df['标签'] == 0).sum()}")
print(f"欺诈交易: {(df['标签'] == 1).sum()}")
print(f"欺诈比例: {df['标签'].mean():.4f}")

# === 特征工程 ===
print("\n=== 特征工程 ===")

# 衍生特征
df["金额偏离比"] = df["交易金额"] / (df["平均交易金额_7d"] + 1)
df["频率异常分数"] = df["交易频率_24h"] / (df["交易频率_24h"].median() + 1)
df["时间异常分数"] = np.abs(df["交易时间小时"] - 14) / 12
df["金额_频率交互"] = df["交易金额"] * df["交易频率_24h"]

feature_cols = ["交易金额", "交易时间小时", "交易频率_24h", "距上次交易_分钟",
                "异地交易标志", "商户类别数_7d", "平均交易金额_7d", "大额交易比例_30d",
                "金额偏离比", "频率异常分数", "时间异常分数", "金额_频率交互"]

print(f"特征数量: {len(feature_cols)}")
print(f"\n正常 vs 欺诈 特征均值对比:")
for col in feature_cols[:8]:
    n_mean = df[df["标签"] == 0][col].mean()
    f_mean = df[df["标签"] == 1][col].mean()
    print(f"  {col}: 正常={n_mean:.2f}, 欺诈={f_mean:.2f}, 比值={f_mean / (n_mean + 0.01):.2f}")

# 标准化
X = df[feature_cols].values
y_true = df["标签"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 多种异常检测算法 ===
print("\n" + "=" * 60)
print("多种异常检测算法对比")
print("=" * 60)

contamination = n_fraud / len(df)

# 算法1: Isolation Forest
print("\n--- 1. Isolation Forest ---")
iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
iso_pred = iso.fit_predict(X_scaled)
iso_pred = np.where(iso_pred == -1, 1, 0)

# 算法2: One-Class SVM
print("--- 2. One-Class SVM ---")
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination * 2)
ocsvm_pred = ocsvm.fit_predict(X_scaled)
ocsvm_pred = np.where(ocsvm_pred == -1, 1, 0)

# 算法3: LOF
print("--- 3. Local Outlier Factor ---")
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
lof_pred = lof.fit_predict(X_scaled)
lof_pred = np.where(lof_pred == -1, 1, 0)

# 算法4: DBSCAN
print("--- 4. DBSCAN ---")
db = DBSCAN(eps=2.0, min_samples=5)
db_labels = db.fit_predict(X_scaled)
db_pred = np.where(db_labels == -1, 1, 0)

# === 评估 ===
print("\n" + "=" * 60)
print("评估结果")
print("=" * 60)

algorithms = {
    "IsolationForest": iso_pred,
    "OneClassSVM": ocsvm_pred,
    "LOF": lof_pred,
    "DBSCAN": db_pred,
}

results = []
for name, pred in algorithms.items():
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    detected = pred.sum()
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    results.append({"算法": name, "检出数": detected, "TP": tp, "FP": fp, "FN": fn,
                     "Precision": p, "Recall": r, "F1": f1})

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format="%.4f"))

# === 最佳算法详细报告 ===
best_name = results_df.loc[results_df["F1"].idxmax(), "算法"]
best_pred = algorithms[best_name]
print(f"\n=== 最佳算法: {best_name} 详细报告 ===")
print(classification_report(y_true, best_pred, target_names=["正常", "欺诈"], zero_division=0))

# === 欺诈交易特征分析 ===
print("=== 欺诈交易特征分析 ===")
fraud_detected = (best_pred == 1) & (y_true == 1)
fraud_missed = (best_pred == 0) & (y_true == 1)

if fraud_detected.sum() > 0:
    print(f"\n成功检测的欺诈交易特征均值:")
    for col in feature_cols[:8]:
        print(f"  {col}: {df.loc[fraud_detected, col].mean():.2f}")

if fraud_missed.sum() > 0:
    print(f"\n漏检的欺诈交易特征均值:")
    for col in feature_cols[:8]:
        print(f"  {col}: {df.loc[fraud_missed, col].mean():.2f}")

print(f"\n=== 总结 ===")
print(f"数据集: {len(df)}笔交易, 欺诈率{contamination:.4f}")
print(f"最佳算法: {best_name}, F1={results_df.loc[results_df['F1'].idxmax(), 'F1']:.4f}")
print("欺诈检测难点: 样本极度不平衡, 欺诈手段不断进化")
print("建议: 结合规则引擎+异常检测+有监督学习, 多层防御")
