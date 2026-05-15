# 数据来源: 模拟电商用户行为漏斗数据
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, scikit-learn>=1.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

steps = ["访问首页", "浏览商品", "加入购物车", "提交订单", "支付成功"]
base_users = 50000
drop_rates = [0.0, 0.45, 0.55, 0.40, 0.25]

user_counts = [base_users]
for i in range(1, len(steps)):
    remaining = int(user_counts[-1] * (1 - drop_rates[i]))
    user_counts.append(remaining)

funnel_df = pd.DataFrame({"步骤": steps, "用户数": user_counts})
funnel_df["步骤转化率"] = funnel_df["用户数"] / funnel_df["用户数"].shift(1)
funnel_df["步骤转化率"] = funnel_df["步骤转化率"].fillna(1.0)
funnel_df["整体转化率"] = funnel_df["用户数"] / funnel_df["用户数"].iloc[0]
funnel_df["流失用户数"] = funnel_df["用户数"].shift(1) - funnel_df["用户数"]
funnel_df["流失用户数"] = funnel_df["流失用户数"].fillna(0).astype(int)
funnel_df["流失率"] = funnel_df["流失用户数"] / funnel_df["用户数"].shift(1)
funnel_df["流失率"] = funnel_df["流失率"].fillna(0.0)

print("=" * 60)
print("漏斗分析结果")
print("=" * 60)
print(funnel_df.to_string(index=False))
print(f"\n整体转化率: {funnel_df['整体转化率'].iloc[-1]:.2%}")

max_loss_idx = funnel_df["流失用户数"].idxmax()
print(f"最大流失环节: {funnel_df.loc[max_loss_idx, '步骤']} (流失 {funnel_df.loc[max_loss_idx, '流失用户数']} 人)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(steps)))
bars = axes[0].barh(steps[::-1], funnel_df["用户数"][::-1], color=colors[::-1])
for bar, count, rate in zip(bars, funnel_df["用户数"][::-1], funnel_df["整体转化率"][::-1]):
    axes[0].text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                 f"{count} ({rate:.1%})", va="center", fontsize=10)
axes[0].set_xlabel("用户数")
axes[0].set_title("转化漏斗（水平柱状图）")

axes[1].plot(steps, funnel_df["用户数"], "o-", color="#2196F3", linewidth=2, markersize=8)
axes[1].fill_between(range(len(steps)), funnel_df["用户数"], alpha=0.15, color="#2196F3")
for i, (s, c) in enumerate(zip(steps, funnel_df["用户数"])):
    axes[1].annotate(f"{c}", (i, c), textcoords="offset points", xytext=(0, 10), ha="center")
axes[1].set_ylabel("用户数")
axes[1].set_title("漏斗流失趋势")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "漏斗分析.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / '漏斗分析.png'}")
