# 数据来源: seaborn 内置数据集 tips
# 分类图是 Seaborn 中用于展示分类变量与数值变量关系的核心图表类型
# 本案例覆盖: barplot, boxplot, violinplot, stripplot, swarmplot, hue 参数, orient 参数, 置信区间
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

tips = sns.load_dataset("tips")
print("数据集前5行:")
print(tips.head())
print(f"\n数据集形状: {tips.shape}")
print(f"列名: {list(tips.columns)}")

# ============================================================
# 1. sns.barplot() — 条形图（均值 + 置信区间）
# ============================================================
# barplot 默认展示每个分类的均值，竖线表示置信区间（默认 95%）
# 置信区间通过 bootstrap 重抽样方法计算，反映均值的估计不确定性

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基础条形图：每天的账单总额均值
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title("基础条形图：每日账单均值（含95%置信区间）")
axes[0, 0].set_xlabel("星期")
axes[0, 0].set_ylabel("账单总额")

# 使用 hue 参数分组：按是否吸烟分组
sns.barplot(data=tips, x="day", y="total_bill", hue="smoker", ax=axes[0, 1])
axes[0, 1].set_title("hue分组：按吸烟状态分组")
axes[0, 1].set_xlabel("星期")
axes[0, 1].set_ylabel("账单总额")

# 修改置信区间：errorbar 参数控制误差线类型和范围
# errorbar=('ci', 68) 表示 68% 置信区间（约1个标准误）
sns.barplot(data=tips, x="day", y="total_bill", errorbar=('ci', 68), ax=axes[1, 0])
axes[1, 0].set_title("68%置信区间条形图")
axes[1, 0].set_xlabel("星期")
axes[1, 0].set_ylabel("账单总额")

# 使用 orient 参数切换方向：水平条形图
sns.barplot(data=tips, y="day", x="total_bill", orient="h", ax=axes[1, 1])
axes[1, 1].set_title("水平条形图（orient='h'）")
axes[1, 1].set_ylabel("星期")
axes[1, 1].set_xlabel("账单总额")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "barplot_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- barplot 要点 ---")
print("1. barplot 默认展示均值，误差线为95%置信区间（bootstrap法）")
print("2. 置信区间越窄，说明均值估计越精确（样本量大或方差小）")
print("3. errorbar=('ci', 68) 可改为68%置信区间，errorbar='sd'展示标准差")
print("4. orient='h' 可绘制水平条形图")

# ============================================================
# 2. sns.boxplot() — 箱线图
# ============================================================
# 箱线图展示数据的五数概括：最小值、Q1、中位数、Q3、最大值
# 箱体表示 IQR（四分位距），须线延伸至1.5*IQR范围内，超出为异常点

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基础箱线图
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title("基础箱线图：每日账单分布")
axes[0, 0].set_xlabel("星期")
axes[0, 0].set_ylabel("账单总额")

# hue 分组箱线图
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker", ax=axes[0, 1])
axes[0, 1].set_title("hue分组箱线图")
axes[0, 1].set_xlabel("星期")
axes[0, 1].set_ylabel("账单总额")

# 水平箱线图
sns.boxplot(data=tips, y="day", x="total_bill", orient="h", ax=axes[1, 0])
axes[1, 0].set_title("水平箱线图")
axes[1, 0].set_ylabel("星期")
axes[1, 0].set_xlabel("账单总额")

# 箱线图 + stripplot 叠加（展示原始数据点）
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[1, 1], whis=[0, 100])
sns.stripplot(data=tips, x="day", y="total_bill", color=".3", size=3, ax=axes[1, 1])
axes[1, 1].set_title("箱线图 + 散点叠加（whis=[0,100]隐藏异常点标记）")
axes[1, 1].set_xlabel("星期")
axes[1, 1].set_ylabel("账单总额")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "boxplot_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- boxplot 要点 ---")
print("1. 箱体: Q1~Q3（25%~75%分位数），中线为Q2（中位数）")
print("2. 须线: 延伸至1.5*IQR范围内的最远数据点")
print("3. 超出须线范围的点标记为异常值（outlier）")
print("4. whis=[0, 100] 可将须线延伸至最小/最大值，隐藏异常点标记")

# ============================================================
# 3. sns.violinplot() — 小提琴图
# ============================================================
# 小提琴图 = 箱线图 + 核密度估计，能同时展示分布的形状和统计摘要

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基础小提琴图
sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title("基础小提琴图")
axes[0, 0].set_xlabel("星期")
axes[0, 0].set_ylabel("账单总额")

# hue 分组 + split 对半分割
# split=True 将两个 hue 组分别绘制在小提琴的左右两侧
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker", split=True, ax=axes[0, 1])
axes[0, 1].set_title("split小提琴图（左右对比吸烟/不吸烟）")
axes[0, 0].set_xlabel("星期")
axes[0, 0].set_ylabel("账单总额")

# inner="quartile" 显示四分位数线
sns.violinplot(data=tips, x="day", y="total_bill", inner="quartile", ax=axes[1, 0])
axes[1, 0].set_title("inner='quartile'：显示四分位线")
axes[1, 0].set_xlabel("星期")
axes[1, 0].set_ylabel("账单总额")

# inner="stick" 显示每个数据点
sns.violinplot(data=tips, x="day", y="total_bill", inner="stick", ax=axes[1, 1])
axes[1, 1].set_title("inner='stick'：显示数据点位置")
axes[1, 1].set_xlabel("星期")
axes[1, 1].set_ylabel("账单总额")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "violinplot_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- violinplot 要点 ---")
print("1. 小提琴图宽度代表数据密度，越宽处数据越密集")
print("2. split=True 可将两个hue组分别展示在小提琴左右两侧")
print("3. inner参数: 'box'(默认箱线图), 'quartile'(四分位线), 'stick'(数据点), None(空心)")

# ============================================================
# 4. sns.stripplot() 与 sns.swarmplot() — 散点分类图
# ============================================================
# stripplot: 带随机抖动的散点图，避免点重叠
# swarmplot: 蜂群图，点不重叠，更直观展示数据密度

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# stripplot 基础用法
sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title("stripplot：抖动散点图")
axes[0, 0].set_xlabel("星期")
axes[0, 0].set_ylabel("账单总额")

# stripplot + hue
sns.stripplot(data=tips, x="day", y="total_bill", hue="smoker", dodge=True, ax=axes[0, 1])
axes[0, 1].set_title("stripplot：dodge=True按hue分组")
axes[0, 1].set_xlabel("星期")
axes[0, 1].set_ylabel("账单总额")

# swarmplot 基础用法
sns.swarmplot(data=tips, x="day", y="total_bill", ax=axes[1, 0])
axes[1, 0].set_title("swarmplot：蜂群图（点不重叠）")
axes[1, 0].set_xlabel("星期")
axes[1, 0].set_ylabel("账单总额")

# swarmplot + hue
sns.swarmplot(data=tips, x="day", y="total_bill", hue="smoker", dodge=True, ax=axes[1, 1])
axes[1, 1].set_title("swarmplot：dodge=True按hue分组")
axes[1, 1].set_xlabel("星期")
axes[1, 1].set_ylabel("账单总额")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "strip_swarm_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- stripplot 与 swarmplot 要点 ---")
print("1. stripplot: 随机抖动避免重叠，jitter参数控制抖动幅度")
print("2. swarmplot: 算法排列避免重叠，更精确展示数据密度")
print("3. dodge=True: hue分组时将不同组沿x轴分开显示")
print("4. swarmplot不适合大数据集（>5000点），会变慢且图形拥挤")

# ============================================================
# 5. 综合对比：同一数据的不同分类图
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.barplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title("barplot：均值+置信区间")

sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 1])
axes[0, 1].set_title("boxplot：五数概括")

sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[0, 2])
axes[0, 2].set_title("violinplot：核密度+箱线")

sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[1, 0])
axes[1, 0].set_title("stripplot：抖动散点")

sns.swarmplot(data=tips, x="day", y="total_bill", ax=axes[1, 1])
axes[1, 1].set_title("swarmplot：蜂群散点")

# boxplot + stripplot 叠加
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[1, 2], whis=[0, 100])
sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[1, 2], color=".3", size=3)
axes[1, 2].set_title("boxplot + stripplot 叠加")

for ax in axes.flat:
    ax.set_xlabel("星期")
    ax.set_ylabel("账单总额")

plt.suptitle("分类图综合对比：同一数据集的五种可视化方式", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "categorical_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 分类图选择指南 ---")
print("barplot:   关注均值比较和置信区间时使用")
print("boxplot:   关注分布摘要和异常值检测时使用")
print("violinplot: 关注分布形状和密度时使用")
print("stripplot:  关注原始数据点分布时使用（大数据集）")
print("swarmplot:  关注精确数据点密度时使用（小数据集）")
