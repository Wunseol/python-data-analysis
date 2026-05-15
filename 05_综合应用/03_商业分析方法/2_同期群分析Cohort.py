# 数据来源: 模拟用户注册与活跃行为数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

n_users = 2000
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-06-30")

reg_dates = pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date, freq="D"), n_users))
users = pd.DataFrame({"user_id": range(n_users), "reg_date": reg_dates})

records = []
for _, row in users.iterrows():
    uid = row["user_id"]
    reg = row["reg_date"]
    active_days = np.random.geometric(0.15, size=1)[0]
    active_days = min(active_days, 60)
    offsets = sorted(np.random.choice(range(active_days + 1), size=min(active_days, np.random.randint(3, 20)), replace=False))
    for off in offsets:
        records.append({"user_id": uid, "active_date": reg + pd.Timedelta(days=int(off))})

df = pd.DataFrame(records)
df["reg_month"] = df["active_date"].map(lambda x: pd.Timestamp(x.year, x.month, 1))

first_active = df.groupby("user_id")["active_date"].min().reset_index()
first_active.columns = ["user_id", "cohort_month"]
first_active["cohort_month"] = first_active["cohort_month"].map(lambda x: pd.Timestamp(x.year, x.month, 1))

df = df.merge(first_active, on="user_id")
df["period"] = ((df["active_date"] - df["cohort_month"]) / pd.Timedelta(days=30)).astype(int)

cohort_data = df.groupby(["cohort_month", "period"])["user_id"].nunique().reset_index()
cohort_data.columns = ["cohort_month", "period", "user_count"]

cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period", values="user_count")

cohort_sizes = df.groupby("cohort_month")["user_id"].nunique()
retention = cohort_pivot.divide(cohort_sizes, axis=0)

print("=" * 60)
print("同期群留存矩阵")
print("=" * 60)
print(retention.round(3).to_string())

fig, ax = plt.subplots(figsize=(12, 8))
mask = retention.notna()
im = ax.imshow(retention.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(retention.columns)))
ax.set_xticklabels([f"第{c}月" for c in retention.columns])
ax.set_yticks(range(len(retention.index)))
ax.set_yticklabels([d.strftime("%Y-%m") for d in retention.index])

for i in range(len(retention.index)):
    for j in range(len(retention.columns)):
        if mask.iloc[i, j]:
            val = retention.iloc[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=8)

plt.colorbar(im, label="留存率")
ax.set_xlabel("距首次活跃月数")
ax.set_ylabel("同期群月份")
ax.set_title("同期群留存热力图")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "同期群分析.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / '同期群分析.png'}")
