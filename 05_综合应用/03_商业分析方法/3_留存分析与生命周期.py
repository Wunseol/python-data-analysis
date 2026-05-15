# 数据来源: 模拟用户留存与消费行为数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

n_users = 5000
start_date = pd.Timestamp("2025-01-01")

records = []
for uid in range(n_users):
    reg = start_date + pd.Timedelta(days=np.random.randint(0, 30))
    n_actions = np.random.poisson(8)
    for _ in range(n_actions):
        offset = int(np.random.exponential(15))
        active_date = reg + pd.Timedelta(days=offset)
        amount = round(np.random.exponential(80) + 10, 2)
        records.append({"user_id": uid, "date": active_date, "amount": amount})

df = pd.DataFrame(records)
first_date = df.groupby("user_id")["date"].min().reset_index()
first_date.columns = ["user_id", "first_date"]
df = df.merge(first_date, on="user_id")
df["day_diff"] = (df["date"] - df["first_date"]).dt.days

retention_data = []
for day in [1, 3, 7, 14, 30, 60, 90]:
    retained = df[df["day_diff"] == day]["user_id"].nunique()
    total = df["user_id"].nunique()
    retention_data.append({"天数": day, "留存用户数": retained, "留存率": retained / total})

retention_df = pd.DataFrame(retention_data)
print("=" * 60)
print("留存率数据")
print("=" * 60)
print(retention_df.to_string(index=False))

print(f"\n次日留存率: {retention_df[retention_df['天数']==1]['留存率'].values[0]:.2%}")
print(f"7日留存率: {retention_df[retention_df['天数']==7]['留存率'].values[0]:.2%}")
print(f"30日留存率: {retention_df[retention_df['天数']==30]['留存率'].values[0]:.2%}")

def retention_model(t, a, b, c):
    return a * np.exp(-b * t) + c

x_data = retention_df["天数"].values.astype(float)
y_data = retention_df["留存率"].values

popt, pcov = curve_fit(retention_model, x_data, y_data, p0=[0.5, 0.05, 0.05], maxfev=5000)
a, b, c = popt
print(f"\n留存衰减模型: R(t) = {a:.4f} * exp(-{b:.4f} * t) + {c:.4f}")

ltv_days = 365
t_pred = np.linspace(0, ltv_days, 1000)
r_pred = retention_model(t_pred, *popt)
ltv_area = np.trapz(r_pred, t_pred)

avg_revenue = df.groupby("user_id")["amount"].sum().mean()
ltv = avg_revenue * (ltv_area / df["day_diff"].mean() if df["day_diff"].mean() > 0 else ltv_area)
print(f"用户生命周期(积分面积): {ltv_area:.1f} 天")
print(f"平均用户消费: {avg_revenue:.2f}")
print(f"估算LTV: {ltv:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(retention_df["天数"], retention_df["留存率"], "o-", color="#2196F3", label="实际留存率")
t_smooth = np.linspace(0, 90, 200)
axes[0].plot(t_smooth, retention_model(t_smooth, *popt), "--", color="#FF5722", label="衰减模型拟合")
axes[0].set_xlabel("天数")
axes[0].set_ylabel("留存率")
axes[0].set_title("留存曲线与衰减模型拟合")
axes[0].legend()
axes[0].grid(alpha=0.3)

t_ltv = np.linspace(0, ltv_days, 1000)
r_ltv = retention_model(t_ltv, *popt)
axes[1].fill_between(t_ltv, r_ltv, alpha=0.3, color="#4CAF50")
axes[1].plot(t_ltv, r_ltv, color="#4CAF50", linewidth=2)
axes[1].set_xlabel("天数")
axes[1].set_ylabel("留存率")
axes[1].set_title(f"用户生命周期价值 (LTV积分面积={ltv_area:.0f}天)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "留存分析.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / '留存分析.png'}")
