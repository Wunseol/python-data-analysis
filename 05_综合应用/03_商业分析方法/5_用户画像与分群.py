# 数据来源: 模拟用户多维度行为特征数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

n_users = 800

profiles = pd.DataFrame({
    "user_id": range(n_users),
    "浏览时长_min": np.random.exponential(30, n_users).round(1),
    "购买频次": np.random.poisson(8, n_users),
    "客单价": (np.random.exponential(120, n_users) + 20).round(2),
    "活跃天数": np.random.poisson(15, n_users),
    "退货率": np.random.beta(2, 10, n_users).round(3),
    "收藏数": np.random.poisson(10, n_users),
    "分享次数": np.random.poisson(3, n_users),
})

profiles["购买频次"] = profiles["购买频次"].clip(lower=1)
profiles["活跃天数"] = profiles["活跃天数"].clip(lower=1)

profiles["消费倾向"] = (profiles["购买频次"] * profiles["客单价"] / profiles["浏览时长_min"]).round(2)
profiles["互动指数"] = (profiles["收藏数"] + profiles["分享次数"]) / profiles["活跃天数"].clip(lower=1)
profiles["忠诚度"] = profiles["活跃天数"] / 30

def tag_user(row):
    tags = []
    if row["消费倾向"] > profiles["消费倾向"].median():
        tags.append("高消费倾向")
    else:
        tags.append("低消费倾向")
    if row["互动指数"] > profiles["互动指数"].median():
        tags.append("高互动")
    else:
        tags.append("低互动")
    if row["忠诚度"] > profiles["忠诚度"].median():
        tags.append("高忠诚")
    else:
        tags.append("低忠诚")
    if row["退货率"] > profiles["退货率"].median():
        tags.append("高退货风险")
    return "|".join(tags)

profiles["标签"] = profiles.apply(tag_user, axis=1)

print("=" * 60)
print("用户画像标签（前10行）")
print("=" * 60)
print(profiles[["user_id", "消费倾向", "互动指数", "忠诚度", "标签"]].head(10).to_string(index=False))

features = ["浏览时长_min", "购买频次", "客单价", "活跃天数", "退货率", "收藏数", "分享次数"]
X = profiles[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
profiles["KMeans分群"] = kmeans.fit_predict(X_scaled)

Z = linkage(X_scaled, method="ward")
profiles["层次分群"] = fcluster(Z, t=4, criterion="maxclust") - 1

cluster_features_k = profiles.groupby("KMeans分群")[features].mean().round(2)
cluster_features_k["用户数"] = profiles["KMeans分群"].value_counts().sort_index().values

print("\n" + "=" * 60)
print("KMeans分群特征对比")
print("=" * 60)
print(cluster_features_k.to_string())

cluster_names = {}
for label in sorted(cluster_features_k.index):
    row = cluster_features_k.loc[label]
    if row["购买频次"] > cluster_features_k["购买频次"].median() and row["客单价"] > cluster_features_k["客单价"].median():
        cluster_names[label] = "高价值用户"
    elif row["活跃天数"] > cluster_features_k["活跃天数"].median() and row["购买频次"] <= cluster_features_k["购买频次"].median():
        cluster_names[label] = "活跃低消用户"
    elif row["退货率"] > cluster_features_k["退货率"].median():
        cluster_names[label] = "高风险用户"
    else:
        cluster_names[label] = "普通用户"

profiles["分群名称"] = profiles["KMeans分群"].map(cluster_names)

print("\n分群命名:")
for label, name in cluster_names.items():
    cnt = (profiles["KMeans分群"] == label).sum()
    print(f"  群{label}: {name} ({cnt}人)")

radar_features = ["购买频次", "客单价", "活跃天数", "收藏数", "分享次数"]
cluster_means = profiles.groupby("KMeans分群")[radar_features].mean()

minmax = MinMaxScaler()
radar_values = minmax.fit_transform(cluster_means)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax_radar = fig.add_subplot(121, polar=True)
angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

colors = ["#FF5722", "#2196F3", "#4CAF50", "#FF9800"]
for i, label in enumerate(sorted(cluster_means.index)):
    values = radar_values[label].tolist()
    values += values[:1]
    ax_radar.plot(angles, values, "o-", color=colors[i], label=cluster_names.get(label, f"群{label}"), linewidth=2)
    ax_radar.fill(angles, values, alpha=0.1, color=colors[i])

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_features)
ax_radar.set_title("用户画像雷达图")
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

ax2 = axes[1]
ax2.set_visible(False)
ax_dendro = fig.add_subplot(122)
dendrogram(Z, truncate_mode="lastp", p=20, ax=ax_dendro, leaf_rotation=45)
ax_dendro.set_title("层次聚类树状图")
ax_dendro.set_xlabel("样本")
ax_dendro.set_ylabel("距离")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "用户画像.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / '用户画像.png'}")
