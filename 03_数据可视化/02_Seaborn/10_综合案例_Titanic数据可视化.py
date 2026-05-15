# 数据来源: seaborn 内置数据集 titanic
# 依赖库最低版本要求: seaborn>=0.13, matplotlib>=3.7, pandas>=2.0

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

titanic = sns.load_dataset("titanic")

print("数据集基本信息:")
print(f"  样本数: {len(titanic)}")
print(f"  特征数: {titanic.shape[1]}")
print(f"  缺失值:\n{titanic.isnull().sum()}\n")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("Titanic 数据可视化综合分析", fontsize=20, fontweight="bold", y=0.98)

# ---- 1. 舱位等级与性别的生存率 (分组柱状图) ----
ax1 = axes[0, 0]
survival_rate = titanic.groupby(["class", "sex"], observed=True)["survived"].mean().reset_index()
survival_rate["survived"] = (survival_rate["survived"] * 100).round(1)
sns.barplot(
    data=survival_rate,
    x="class",
    y="survived",
    hue="sex",
    palette="Set2",
    ax=ax1,
)
ax1.set_title("各舱位等级与性别的生存率", fontsize=14, fontweight="bold")
ax1.set_xlabel("舱位等级", fontsize=12)
ax1.set_ylabel("生存率 (%)", fontsize=12)
ax1.legend(title="性别", fontsize=10)
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)

# ---- 2. 生存情况的年龄分布 (重叠直方图 + KDE) ----
ax2 = axes[0, 1]
survived = titanic[titanic["survived"] == 1]["age"].dropna()
not_survived = titanic[titanic["survived"] == 0]["age"].dropna()
ax2.hist(survived, bins=30, alpha=0.5, label="幸存", color="#2ecc71", density=True)
ax2.hist(not_survived, bins=30, alpha=0.5, label="未幸存", color="#e74c3c", density=True)
survived.plot.kde(ax=ax2, color="#27ae60", linewidth=2)
not_survived.plot.kde(ax=ax2, color="#c0392b", linewidth=2)
ax2.set_title("生存情况的年龄分布", fontsize=14, fontweight="bold")
ax2.set_xlabel("年龄", fontsize=12)
ax2.set_ylabel("密度", fontsize=12)
ax2.legend(fontsize=11)

# ---- 3. 各舱位等级的票价分布 (箱线图) ----
ax3 = axes[1, 0]
sns.boxplot(
    data=titanic,
    x="class",
    y="fare",
    hue="class",
    palette="Blues",
    ax=ax3,
    legend=False,
)
ax3.set_title("各舱位等级的票价分布", fontsize=14, fontweight="bold")
ax3.set_xlabel("舱位等级", fontsize=12)
ax3.set_ylabel("票价", fontsize=12)

# ---- 4. 登船港口分析 (计数图) ----
ax4 = axes[1, 1]
port_map = {"C": "瑟堡", "Q": "皇后镇", "S": "南安普顿"}
titanic["embark_town_cn"] = titanic["embarked"].map(port_map)
sns.countplot(
    data=titanic,
    x="embark_town_cn",
    hue="survived",
    palette={0: "#e74c3c", 1: "#2ecc71"},
    ax=ax4,
)
ax4.set_title("登船港口与生存情况", fontsize=14, fontweight="bold")
ax4.set_xlabel("登船港口", fontsize=12)
ax4.set_ylabel("人数", fontsize=12)
ax4.legend(title="是否幸存", labels=["未幸存", "幸存"], fontsize=10)
for container in ax4.containers:
    ax4.bar_label(container, fontsize=9, padding=3)

# ---- 5. 家庭规模对生存率的影响 (柱状图) ----
ax5 = axes[2, 0]
titanic["family_size"] = titanic["sibsp"] + titanic["parch"] + 1
family_survival = titanic.groupby("family_size")["survived"].mean().reset_index()
family_survival["survived"] = (family_survival["survived"] * 100).round(1)
bar_colors = ["#e74c3c" if v < 40 else "#f39c12" if v < 60 else "#2ecc71" for v in family_survival["survived"]]
bars = ax5.bar(
    family_survival["family_size"],
    family_survival["survived"],
    color=bar_colors,
    edgecolor="white",
)
ax5.set_title("家庭规模对生存率的影响", fontsize=14, fontweight="bold")
ax5.set_xlabel("家庭规模 (含本人)", fontsize=12)
ax5.set_ylabel("生存率 (%)", fontsize=12)
ax5.set_xticks(family_survival["family_size"])
for bar, val in zip(bars, family_survival["survived"]):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# ---- 6. 数值特征相关性热力图 ----
ax6 = axes[2, 1]
numeric_cols = titanic[["survived", "pclass", "age", "sibsp", "parch", "fare"]].dropna()
corr = numeric_cols.corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax6,
    annot_kws={"fontsize": 10},
)
ax6.set_title("数值特征相关性热力图", fontsize=14, fontweight="bold")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right")
ax6.set_yticklabels(ax6.get_yticklabels(), rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_dir = Path(__file__).parent
output_path = output_dir / "titanic_visualization.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"图表已保存至: {output_path}")

plt.show()
