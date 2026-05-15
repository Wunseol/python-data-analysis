# 数据来源: 模拟电商用户交易数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

n_users = 1000
reference_date = pd.Timestamp("2025-12-31")

records = []
for uid in range(n_users):
    n_orders = max(1, np.random.poisson(5))
    for _ in range(n_orders):
        days_ago = np.random.exponential(60)
        date = reference_date - pd.Timedelta(days=int(days_ago))
        amount = round(np.random.exponential(150) + 20, 2)
        records.append({"user_id": uid, "order_date": date, "amount": amount})

df = pd.DataFrame(records)

rfm = df.groupby("user_id").agg(
    Recency=("order_date", lambda x: (reference_date - x.max()).days),
    Frequency=("order_date", "count"),
    Monetary=("amount", "sum")
).reset_index()

print("=" * 60)
print("RFM原始数据（前10行）")
print("=" * 60)
print(rfm.head(10).to_string(index=False))

r_labels = list(range(5, 0, -1))
f_labels = list(range(1, 6))
m_labels = list(range(1, 6))

rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=r_labels, duplicates="drop").astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"], 5, labels=f_labels, duplicates="drop").astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=m_labels, duplicates="drop").astype(int)
rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

def rfm_level(score):
    if score >= 13:
        return "重要价值用户"
    elif score >= 10:
        return "重要发展用户"
    elif score >= 7:
        return "一般价值用户"
    elif score >= 4:
        return "一般发展用户"
    else:
        return "流失预警用户"

rfm["价值等级"] = rfm["RFM_total"].apply(rfm_level)

print("\n" + "=" * 60)
print("RFM评分结果（前10行）")
print("=" * 60)
print(rfm.head(10).to_string(index=False))

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm["聚类标签"] = kmeans.fit_predict(rfm_scaled)

cluster_stats = rfm.groupby("聚类标签")[["Recency", "Frequency", "Monetary"]].mean()
cluster_counts = rfm["聚类标签"].value_counts().sort_index()
cluster_stats["用户数"] = cluster_counts.values

cluster_names = {}
for label in sorted(cluster_stats.index):
    r = cluster_stats.loc[label, "Recency"]
    f = cluster_stats.loc[label, "Frequency"]
    m = cluster_stats.loc[label, "Monetary"]
    if r < cluster_stats["Recency"].median() and f > cluster_stats["Frequency"].median():
        cluster_names[label] = "高价值活跃"
    elif r < cluster_stats["Recency"].median() and m > cluster_stats["Monetary"].median():
        cluster_names[label] = "高价值沉默"
    elif f > cluster_stats["Frequency"].median():
        cluster_names[label] = "高频低额"
    elif r > cluster_stats["Recency"].median():
        cluster_names[label] = "流失风险"
    else:
        cluster_names[label] = "普通用户"

rfm["聚类名称"] = rfm["聚类标签"].map(cluster_names)

print("\n" + "=" * 60)
print("KMeans聚类分层结果")
print("=" * 60)
for label in sorted(cluster_stats.index):
    name = cluster_names[label]
    row = cluster_stats.loc[label]
    print(f"  {name}: {int(row['用户数'])}人 | R={row['Recency']:.1f} F={row['Frequency']:.1f} M={row['Monetary']:.1f}")

level_stats = rfm.groupby("价值等级").agg(
    用户数=("user_id", "count"),
    平均消费=("Monetary", "mean"),
    平均频次=("Frequency", "mean")
).round(2)
print("\n" + "=" * 60)
print("各价值等级特征")
print("=" * 60)
print(level_stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

level_counts = rfm["价值等级"].value_counts()
colors = ["#FF5722", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
axes[0].bar(level_counts.index, level_counts.values, color=colors)
axes[0].set_title("RFM价值等级分布")
axes[0].set_ylabel("用户数")
for i, v in enumerate(level_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center")

scatter = axes[1].scatter(rfm["Recency"], rfm["Monetary"], c=rfm["聚类标签"], cmap="Set1", alpha=0.5, s=10)
axes[1].set_xlabel("Recency (天)")
axes[1].set_ylabel("Monetary (元)")
axes[1].set_title("KMeans聚类结果 (R-M散点)")
plt.colorbar(scatter, label="聚类标签")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "RFM分析.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / 'RFM分析.png'}")
