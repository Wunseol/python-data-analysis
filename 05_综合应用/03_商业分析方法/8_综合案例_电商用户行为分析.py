# 数据来源: 模拟电商用户全链路行为数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import curve_fit
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("  电商用户行为综合分析报告")
print("=" * 70)

# ========== 1. 模拟数据生成 ==========
n_users = 3000
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-06-30")

reg_dates = pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date, freq="D"), n_users))
users = pd.DataFrame({"user_id": range(n_users), "reg_date": reg_dates})

funnel_steps = ["浏览", "加购", "下单", "支付"]
funnel_probs = [1.0, 0.55, 0.45, 0.70]

records = []
for _, u in users.iterrows():
    uid = u["user_id"]
    reg = u["reg_date"]
    n_visits = max(1, np.random.poisson(6))
    for _ in range(n_visits):
        offset = int(np.random.exponential(20))
        visit_date = reg + pd.Timedelta(days=offset)
        if visit_date > end_date:
            continue
        step = 0
        for s in range(1, len(funnel_steps)):
            if np.random.random() < funnel_probs[s]:
                step = s
            else:
                break
        amount = round(np.random.exponential(120) + 30, 2) if step >= 2 else 0
        records.append({
            "user_id": uid, "date": visit_date,
            "最远步骤": funnel_steps[step], "步骤序号": step, "amount": amount
        })

df = pd.DataFrame(records)
df["reg_month"] = df["date"].map(lambda x: pd.Timestamp(x.year, x.month, 1))

print(f"\n数据概览: {len(df)} 条行为记录, {df['user_id'].nunique()} 位用户")

# ========== 2. 漏斗分析 ==========
print("\n" + "=" * 70)
print("  一、漏斗分析")
print("=" * 70)

step_counts = df.groupby("步骤序号")["user_id"].nunique()
step_names = [funnel_steps[i] for i in step_counts.index]
funnel_result = pd.DataFrame({"步骤": step_names, "用户数": step_counts.values})
funnel_result["步骤转化率"] = funnel_result["用户数"] / funnel_result["用户数"].shift(1)
funnel_result["步骤转化率"] = funnel_result["步骤转化率"].fillna(1.0)
funnel_result["整体转化率"] = funnel_result["用户数"] / funnel_result["用户数"].iloc[0]

print(funnel_result.to_string(index=False))
print(f"整体转化率: {funnel_result['整体转化率'].iloc[-1]:.2%}")

max_drop_idx = funnel_result["步骤转化率"][1:].idxmin()
print(f"最大流失环节: {funnel_result.loc[max_drop_idx, '步骤']}")

# ========== 3. 同期群分析 ==========
print("\n" + "=" * 70)
print("  二、同期群分析")
print("=" * 70)

first_active = df.groupby("user_id")["date"].min().reset_index()
first_active.columns = ["user_id", "cohort_month"]
first_active["cohort_month"] = first_active["cohort_month"].map(lambda x: pd.Timestamp(x.year, x.month, 1))

df_cohort = df.merge(first_active, on="user_id")
df_cohort["period"] = ((df_cohort["date"] - df_cohort["cohort_month"]) / pd.Timedelta(days=30)).astype(int)

cohort_data = df_cohort.groupby(["cohort_month", "period"])["user_id"].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period", values="user_count")
cohort_sizes = df_cohort.groupby("cohort_month")["user_id"].nunique()
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0)

print("同期群留存矩阵（前3个月）:")
print(retention_matrix.iloc[:3, :4].round(3).to_string())

# ========== 4. RFM分层 ==========
print("\n" + "=" * 70)
print("  四、RFM用户价值分层")
print("=" * 70)

paid = df[df["amount"] > 0].copy()
reference_date = df["date"].max()

rfm = paid.groupby("user_id").agg(
    Recency=("date", lambda x: (reference_date - x.max()).days),
    Frequency=("date", "count"),
    Monetary=("amount", "sum")
).reset_index()

r_labels = list(range(5, 0, -1))
f_labels = list(range(1, 6))
m_labels = list(range(1, 6))

rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=r_labels, duplicates="drop").astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"], 5, labels=f_labels, duplicates="drop").astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"], 5, labels=m_labels, duplicates="drop").astype(int)
rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

def rfm_level(score):
    if score >= 13:
        return "重要价值"
    elif score >= 10:
        return "重要发展"
    elif score >= 7:
        return "一般价值"
    elif score >= 4:
        return "一般发展"
    else:
        return "流失预警"

rfm["价值等级"] = rfm["RFM_total"].apply(rfm_level)

level_stats = rfm.groupby("价值等级").agg(用户数=("user_id", "count"), 平均消费=("Monetary", "mean")).round(2)
print(level_stats.to_string())

# ========== 5. 留存分析 ==========
print("\n" + "=" * 70)
print("  五、留存分析")
print("=" * 70)

df_ret = df.merge(first_active, on="user_id", suffixes=("", "_first"))
if "cohort_month" not in df_ret.columns:
    df_ret = df_ret.rename(columns={"cohort_month": "first_date"})
df_ret["day_diff"] = (df_ret["date"] - df_ret["first_date"]).dt.days

retention_points = []
for day in [1, 7, 30]:
    retained = df_ret[df_ret["day_diff"] == day]["user_id"].nunique()
    total = df_ret["user_id"].nunique()
    retention_points.append({"天数": day, "留存率": retained / total if total > 0 else 0})

ret_df = pd.DataFrame(retention_points)
for _, row in ret_df.iterrows():
    print(f"  {row['天数']}日留存率: {row['留存率']:.2%}")

x_ret = np.array([0] + ret_df["天数"].tolist()).astype(float)
y_ret = np.array([1.0] + ret_df["留存率"].tolist())

def retention_model(t, a, b, c):
    return a * np.exp(-b * t) + c

try:
    popt, _ = curve_fit(retention_model, x_ret, y_ret, p0=[0.5, 0.05, 0.05], maxfev=5000)
    t_ltv = np.linspace(0, 365, 1000)
    r_ltv = retention_model(t_ltv, *popt)
    ltv_days = np.trapz(r_ltv, t_ltv)
    avg_rev = paid.groupby("user_id")["amount"].sum().mean()
    print(f"  估算LTV: {avg_rev * ltv_days / 90:.2f} 元")
except Exception:
    popt = [0.5, 0.05, 0.05]
    ltv_days = 60

# ========== 6. 用户画像 ==========
print("\n" + "=" * 70)
print("  六、用户画像")
print("=" * 70)

user_features = df.groupby("user_id").agg(
    浏览次数=("date", "count"),
    最远步骤=("步骤序号", "max"),
    总消费=("amount", "sum"),
    活跃天数=("date", "nunique"),
    首次活跃=("date", "min"),
    末次活跃=("date", "max"),
).reset_index()

user_features["客单价"] = user_features["总消费"] / user_features["浏览次数"].clip(lower=1)
user_features["活跃跨度"] = (user_features["末次活跃"] - user_features["首次活跃"]).dt.days

feat_cols = ["浏览次数", "最远步骤", "总消费", "活跃天数", "客单价", "活跃跨度"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_features[feat_cols])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
user_features["分群"] = kmeans.fit_predict(X_scaled)

cluster_stats = user_features.groupby("分群")[feat_cols].mean().round(2)
cluster_stats["用户数"] = user_features["分群"].value_counts().sort_index().values

cluster_names = {}
for label in sorted(cluster_stats.index):
    row = cluster_stats.loc[label]
    if row["总消费"] > cluster_stats["总消费"].median() and row["活跃天数"] > cluster_stats["活跃天数"].median():
        cluster_names[label] = "高价值活跃"
    elif row["总消费"] > cluster_stats["总消费"].median():
        cluster_names[label] = "高价值沉默"
    elif row["活跃天数"] > cluster_stats["活跃天数"].median():
        cluster_names[label] = "活跃低消"
    else:
        cluster_names[label] = "低价值用户"

user_features["分群名称"] = user_features["分群"].map(cluster_names)

for label in sorted(cluster_stats.index):
    name = cluster_names[label]
    cnt = cluster_stats.loc[label, "用户数"]
    print(f"  {name}: {int(cnt)}人 | 消费={cluster_stats.loc[label, '总消费']:.0f} 活跃={cluster_stats.loc[label, '活跃天数']:.0f}天")

# ========== 7. 综合可视化 ==========
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 漏斗图
colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(funnel_result)))
axes[0, 0].barh(funnel_result["步骤"][::-1], funnel_result["用户数"][::-1], color=colors[::-1])
for i, (cnt, rate) in enumerate(zip(funnel_result["用户数"][::-1], funnel_result["整体转化率"][::-1])):
    axes[0, 0].text(cnt + 50, i, f"{cnt} ({rate:.1%})", va="center", fontsize=9)
axes[0, 0].set_title("转化漏斗")

# 同期群热力图
mask = retention_matrix.notna()
ret_sub = retention_matrix.iloc[:6, :5]
im = axes[0, 1].imshow(ret_sub.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
axes[0, 1].set_xticks(range(len(ret_sub.columns)))
axes[0, 1].set_xticklabels([f"第{c}月" for c in ret_sub.columns], fontsize=8)
axes[0, 1].set_yticks(range(len(ret_sub.index)))
axes[0, 1].set_yticklabels([d.strftime("%m月") for d in ret_sub.index], fontsize=8)
for i in range(len(ret_sub.index)):
    for j in range(len(ret_sub.columns)):
        if ret_sub.iloc[i, j] == ret_sub.iloc[i, j]:
            val = ret_sub.iloc[i, j]
            axes[0, 1].text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7,
                           color="white" if val > 0.5 else "black")
axes[0, 1].set_title("同期群留存热力图")

# 留存曲线
t_smooth = np.linspace(0, 60, 200)
axes[0, 2].plot(x_ret, y_ret, "o", color="#2196F3", markersize=8, label="实际留存")
axes[0, 2].plot(t_smooth, retention_model(t_smooth, *popt), "--", color="#FF5722", label="模型拟合")
axes[0, 2].set_xlabel("天数")
axes[0, 2].set_ylabel("留存率")
axes[0, 2].set_title("留存曲线")
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# RFM分布
level_counts = rfm["价值等级"].value_counts()
rfm_colors = ["#FF5722", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
axes[1, 0].bar(level_counts.index, level_counts.values, color=rfm_colors[:len(level_counts)])
axes[1, 0].set_title("RFM价值等级分布")
axes[1, 0].tick_params(axis="x", rotation=30)
for i, v in enumerate(level_counts.values):
    axes[1, 0].text(i, v + 5, str(v), ha="center")

# 用户画像雷达图
radar_feats = ["浏览次数", "总消费", "活跃天数", "客单价", "活跃跨度"]
cluster_means = user_features.groupby("分群")[radar_feats].mean()
minmax = MinMaxScaler()
radar_vals = minmax.fit_transform(cluster_means)

ax_radar = fig.add_subplot(235, polar=True)
angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
angles += angles[:1]
radar_colors = ["#FF5722", "#2196F3", "#4CAF50", "#FF9800"]
for i, label in enumerate(sorted(cluster_means.index)):
    values = radar_vals[label].tolist()
    values += values[:1]
    ax_radar.plot(angles, values, "o-", color=radar_colors[i], label=cluster_names[label], linewidth=2)
    ax_radar.fill(angles, values, alpha=0.1, color=radar_colors[i])
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_feats, fontsize=8)
ax_radar.set_title("用户画像雷达图", pad=20)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

# 分群饼图
group_counts = user_features["分群名称"].value_counts()
axes[1, 2].pie(group_counts.values, labels=group_counts.index, autopct="%1.1f%%",
               colors=radar_colors[:len(group_counts)], startangle=90)
axes[1, 2].set_title("用户分群占比")

plt.suptitle("电商用户行为综合分析报告", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "综合分析报告.png", dpi=150, bbox_inches="tight")
plt.close()

# ========== 8. 综合报告 ==========
print("\n" + "=" * 70)
print("  综合分析摘要")
print("=" * 70)
print(f"  总用户数: {n_users}")
print(f"  付费用户数: {len(rfm)}")
print(f"  漏斗整体转化率: {funnel_result['整体转化率'].iloc[-1]:.2%}")
print(f"  次日留存率: {ret_df[ret_df['天数']==1]['留存率'].values[0]:.2%}")
print(f"  7日留存率: {ret_df[ret_df['天数']==7]['留存率'].values[0]:.2%}")
print(f"  高价值用户占比: {(rfm['价值等级']=='重要价值').mean():.2%}")
print(f"  流失预警用户占比: {(rfm['价值等级']=='流失预警').mean():.2%}")
print(f"\n图表已保存至: {OUTPUT_DIR / '综合分析报告.png'}")
