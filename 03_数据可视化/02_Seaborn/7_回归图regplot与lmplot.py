# 数据来源: seaborn 内置数据集 tips, anscombe
# 回归图用于展示两个数值变量之间的线性（或非线性）关系
# 本案例覆盖: regplot, lmplot, scatter_kws, line_kws, order, logistic回归, 残差图, col/row参数
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

tips = sns.load_dataset("tips")
print("tips 数据集前5行:")
print(tips.head())

# ============================================================
# 1. sns.regplot() — 基础回归图
# ============================================================
# regplot 在散点图上叠加回归线与置信区间带
# 它是轴级函数（axes-level），可以直接指定 ax 参数

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基础回归图
sns.regplot(data=tips, x="total_bill", y="tip", ax=axes[0, 0])
axes[0, 0].set_title("基础回归图：账单总额 vs 小费")
axes[0, 0].set_xlabel("账单总额")
axes[0, 0].set_ylabel("小费")

# scatter_kws 和 line_kws 自定义样式
# scatter_kws 传递给散点的 matplotlib scatter 参数
# line_kws 传递给回归线的 matplotlib plot 参数
sns.regplot(
    data=tips, x="total_bill", y="tip",
    scatter_kws={"color": "steelblue", "alpha": 0.4, "s": 30},
    line_kws={"color": "red", "linewidth": 2, "linestyle": "--"},
    ax=axes[0, 1]
)
axes[0, 1].set_title("自定义样式：scatter_kws + line_kws")
axes[0, 1].set_xlabel("账单总额")
axes[0, 1].set_ylabel("小费")

# order 参数：多项式回归（非线性拟合）
# order=2 拟合二次多项式，order=3 拟合三次多项式
sns.regplot(data=tips, x="total_bill", y="tip", order=2, ax=axes[1, 0])
axes[1, 0].set_title("order=2：二次多项式回归")
axes[1, 0].set_xlabel("账单总额")
axes[1, 0].set_ylabel("小费")

# 关闭置信区间带
sns.regplot(data=tips, x="total_bill", y="tip", ci=None, ax=axes[1, 1])
axes[1, 1].set_title("ci=None：关闭置信区间带")
axes[1, 1].set_xlabel("账单总额")
axes[1, 1].set_ylabel("小费")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "regplot_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- regplot 要点 ---")
print("1. regplot 是轴级函数，可以在指定 ax 上绘制")
print("2. scatter_kws 控制散点样式，line_kws 控制回归线样式")
print("3. order 参数控制多项式阶数（默认1=线性）")
print("4. ci 参数控制置信区间（默认95%），ci=None关闭")

# ============================================================
# 2. sns.lmplot() — 图级回归图（支持分面）
# ============================================================
# lmplot 是图级函数（figure-level），结合了 regplot 和 FacetGrid
# 支持 col/row/hue 参数进行分面，适合多组比较

fig1 = sns.lmplot(data=tips, x="total_bill", y="tip", hue="smoker", height=5, aspect=1.2)
fig1.figure.suptitle("lmplot：按吸烟状态分组回归", y=1.02)
fig1.figure.savefig(Path(__file__).parent / "lmplot_hue.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- lmplot hue 参数 ---")
print("hue 参数按颜色分组，每组分别拟合回归线")

# col 参数：按列分面
fig2 = sns.lmplot(
    data=tips, x="total_bill", y="tip",
    col="smoker", height=5, aspect=1
)
fig2.figure.suptitle("lmplot：col='smoker' 按列分面", y=1.02)
fig2.figure.savefig(Path(__file__).parent / "lmplot_col.png", dpi=150, bbox_inches='tight')
plt.show()

# col + row 参数：二维分面
fig3 = sns.lmplot(
    data=tips, x="total_bill", y="tip",
    col="smoker", row="time", height=4, aspect=1
)
fig3.figure.suptitle("lmplot：col='smoker' + row='time' 二维分面", y=1.02)
fig3.figure.savefig(Path(__file__).parent / "lmplot_col_row.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- lmplot col/row 参数 ---")
print("col: 按列方向分面（水平排列多个子图）")
print("row: 按行方向分面（垂直排列多个子图）")
print("col+row: 二维分面网格，每个组合一个子图")

# ============================================================
# 3. regplot 与 lmplot 的区别
# ============================================================
print("\n--- regplot vs lmplot ---")
print("regplot: 轴级函数，灵活指定ax，适合嵌入自定义布局")
print("lmplot:  图级函数，自动创建FacetGrid，支持col/row/hue分面")
print("regplot: 不支持 col/row 参数")
print("lmplot:  支持 col/row/hue 参数，适合多组比较")
print("两者共享相同的回归拟合逻辑（参数如 order, ci, logistic 等）")

# ============================================================
# 4. logistic 回归图
# ============================================================
# 当 y 为二元变量时，可使用 logistic=True 拟合逻辑回归
# 这里用 tips 数据构造一个二元变量：小费是否超过3元

tips["tip_above_3"] = (tips["tip"] > 3).astype(int)
print("\n构造二元变量 tip_above_3（小费>3元）:")
print(tips[["total_bill", "tip", "tip_above_3"]].head(10))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 线性回归（不适合二元响应变量）
sns.regplot(data=tips, x="total_bill", y="tip_above_3", ax=axes[0])
axes[0].set_title("线性回归（不适合二元变量）")
axes[0].set_xlabel("账单总额")
axes[0].set_ylabel("小费>3元 (0/1)")

# logistic 回归（适合二元响应变量）
sns.regplot(data=tips, x="total_bill", y="tip_above_3", logistic=True, ax=axes[1])
axes[1].set_title("logistic回归（适合二元变量）")
axes[1].set_xlabel("账单总额")
axes[1].set_ylabel("小费>3元 (0/1)")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "logistic_regplot.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- logistic 回归要点 ---")
print("1. 当因变量为二元（0/1）时，线性回归可能预测出<0或>1的值")
print("2. logistic=True 使用逻辑回归，预测值限制在[0,1]区间")
print("3. 回归线呈S形（sigmoid曲线），更合理地拟合二元数据")

# ============================================================
# 5. 残差图概念
# ============================================================
# 残差 = 观测值 - 预测值，残差图用于检验回归模型假设
# Seaborn 没有直接的残差图函数，但可以手动计算并绘制

from sklearn.linear_model import LinearRegression

X = tips[["total_bill"]].values
y = tips["tip"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

print(f"\n线性回归结果: 斜率={model.coef_[0]:.4f}, 截距={model.intercept_:.4f}")
print(f"残差均值: {residuals.mean():.6f}（应接近0）")
print(f"残差标准差: {residuals.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 残差 vs 拟合值
axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title("残差图：残差 vs 拟合值")
axes[0].set_xlabel("拟合值")
axes[0].set_ylabel("残差")

# 残差直方图
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_title("残差分布直方图")
axes[1].set_xlabel("残差")
axes[1].set_ylabel("频数")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "residual_plot.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 残差图要点 ---")
print("1. 残差图用于检验回归假设：线性性、等方差性、正态性")
print("2. 理想残差图：残差随机散布在0附近，无趋势")
print("3. 若残差呈漏斗形 → 方差不齐（异方差）")
print("4. 若残差呈曲线 → 非线性关系（考虑 order>1）")

# ============================================================
# 6. Anscombe 四重奏 — 回归前先看散点图
# ============================================================
# Anscombe 四组数据有几乎相同的统计摘要，但图形完全不同
# 说明仅靠统计量不够，必须可视化

anscombe = sns.load_dataset("anscombe")
print("\nAnscombe 四重奏数据集:")
print(anscombe.head())

g = sns.lmplot(
    data=anscombe, x="x", y="y",
    col="dataset", col_wrap=2,
    height=4, aspect=1,
    scatter_kws={"s": 50, "alpha": 0.8},
    line_kws={"color": "red"}
)
g.figure.suptitle("Anscombe四重奏：相同统计量，不同图形", y=1.02)
g.figure.savefig(Path(__file__).parent / "anscombe_quartet.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Anscombe 四重奏 ---")
print("四组数据有几乎相同的均值、方差、相关系数和回归线")
print("但散点图形状完全不同，强调可视化的重要性")
