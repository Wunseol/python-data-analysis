# 数据来源: seaborn 内置数据集 tips, iris
# 依赖库最低版本要求: seaborn>=0.13, matplotlib>=3.7, pandas>=2.0
# 关系图（relational plots）是 Seaborn 中用于展示两个数值变量之间关系的核心图表类型
# 本案例覆盖: scatterplot, lineplot, relplot, hue/size/style参数, col/row分面, 自定义样式
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

print("=== tips 数据集 ===")
print(tips.head())
print(f"形状: {tips.shape}")
print(f"列名: {list(tips.columns)}")

print("\n=== iris 数据集 ===")
print(iris.head())
print(f"形状: {iris.shape}")
print(f"列名: {list(iris.columns)}")

# ============================================================
# 1. sns.scatterplot() — 散点图
# ============================================================
# scatterplot 是最基础的关系图，用点表示两个数值变量的观测值
# 核心语义参数: hue（颜色）, size（大小）, style（标记形状）

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基础散点图
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[0, 0])
axes[0, 0].set_title("基础散点图：账单总额 vs 小费")
axes[0, 0].set_xlabel("账单总额")
axes[0, 0].set_ylabel("小费")

# hue 参数：按颜色区分分类变量
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker", ax=axes[0, 1])
axes[0, 1].set_title("hue='smoker'：按吸烟状态着色")
axes[0, 1].set_xlabel("账单总额")
axes[0, 1].set_ylabel("小费")

# hue + size 参数：同时用颜色和点大小区分
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker", size="size", ax=axes[1, 0])
axes[1, 0].set_title("hue='smoker' + size='size'：颜色+大小")
axes[1, 0].set_xlabel("账单总额")
axes[1, 0].set_ylabel("小费")

# hue + style 参数：同时用颜色和标记形状区分
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker", style="time", ax=axes[1, 1])
axes[1, 1].set_title("hue='smoker' + style='time'：颜色+形状")
axes[1, 1].set_xlabel("账单总额")
axes[1, 1].set_ylabel("小费")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "scatterplot_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- scatterplot 要点 ---")
print("1. scatterplot 是轴级函数（axes-level），可指定 ax 参数嵌入自定义布局")
print("2. hue: 按颜色区分分类变量，自动生成图例")
print("3. size: 按点大小区分数值/分类变量，映射连续值时自动生成尺寸图例")
print("4. style: 按标记形状区分分类变量（圆、方、三角等）")
print("5. 三个语义参数可组合使用，但不宜过多以免图形混乱")

# ============================================================
# 2. sns.scatterplot() — iris 数据集多变量探索
# ============================================================
# iris 数据集有4个数值特征 + 1个分类特征（species），非常适合演示多语义映射

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# hue 区分鸢尾花品种
sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species", ax=axes[0])
axes[0].set_title("鸢尾花：花萼长度 vs 花瓣长度（hue=品种）")
axes[0].set_xlabel("花萼长度 (cm)")
axes[0].set_ylabel("花瓣长度 (cm)")

# hue + size + style 全语义映射
sns.scatterplot(
    data=tips, x="total_bill", y="tip",
    hue="day", size="size", style="time",
    ax=axes[1]
)
axes[1].set_title("全语义映射：hue=day, size=size, style=time")
axes[1].set_xlabel("账单总额")
axes[1].set_ylabel("小费")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "scatterplot_iris.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- iris 散点图要点 ---")
print("1. iris 三个品种在花瓣长度上区分明显，setosa 完全可线性分离")
print("2. 全语义映射信息丰富但可能难以阅读，建议最多组合两个语义参数")

# ============================================================
# 3. sns.lineplot() — 折线图
# ============================================================
# lineplot 用于展示数值变量随另一个数值变量的变化趋势
# 默认对重复 x 值取均值，并绘制置信区间带（默认95%）

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基础折线图：账单总额 vs 小费
# 当 x 有重复值时，lineplot 自动聚合（取均值）并显示置信区间
sns.lineplot(data=tips, x="size", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title("基础折线图：用餐人数 vs 账单均值（含95%置信区间）")
axes[0, 0].set_xlabel("用餐人数")
axes[0, 0].set_ylabel("账单总额")

# hue 参数：按分类变量分组绘制多条线
sns.lineplot(data=tips, x="size", y="total_bill", hue="smoker", ax=axes[0, 1])
axes[0, 1].set_title("hue='smoker'：按吸烟状态分组")
axes[0, 1].set_xlabel("用餐人数")
axes[0, 1].set_ylabel("账单总额")

# style 参数：不同线型区分
sns.lineplot(data=tips, x="size", y="total_bill", hue="smoker", style="smoker", ax=axes[1, 0])
axes[1, 0].set_title("hue + style：颜色+线型双重编码")
axes[1, 0].set_xlabel("用餐人数")
axes[1, 0].set_ylabel("账单总额")

# markers=True：在数据点处显示标记
sns.lineplot(
    data=tips, x="size", y="total_bill",
    hue="smoker", style="smoker", markers=True,
    ax=axes[1, 1]
)
axes[1, 1].set_title("markers=True：数据点处显示标记")
axes[1, 1].set_xlabel("用餐人数")
axes[1, 1].set_ylabel("账单总额")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "lineplot_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- lineplot 要点 ---")
print("1. lineplot 对重复 x 值自动聚合（默认取均值），并绘制置信区间带")
print("2. 置信区间通过 bootstrap 方法计算，阴影带越窄说明估计越精确")
print("3. errorbar=('ci', 68) 可改为68%置信区间，errorbar='sd'展示标准差")
print("4. hue 按颜色分组，style 按线型分组，markers=True 显示数据点标记")
print("5. 适合展示趋势变化，尤其是时间序列或有序离散变量")

# ============================================================
# 4. sns.lineplot() — 置信区间控制
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 默认 95% 置信区间
sns.lineplot(data=tips, x="size", y="total_bill", ax=axes[0])
axes[0].set_title("默认 95% 置信区间")
axes[0].set_xlabel("用餐人数")
axes[0].set_ylabel("账单总额")

# 68% 置信区间（约1个标准误）
sns.lineplot(data=tips, x="size", y="total_bill", errorbar=('ci', 68), ax=axes[1])
axes[1].set_title("68% 置信区间")
axes[1].set_xlabel("用餐人数")
axes[1].set_ylabel("账单总额")

# 标准差代替置信区间
sns.lineplot(data=tips, x="size", y="total_bill", errorbar='sd', ax=axes[2])
axes[2].set_title("errorbar='sd'：展示标准差")
axes[2].set_xlabel("用餐人数")
axes[2].set_ylabel("账单总额")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "lineplot_errorbar.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- lineplot 置信区间要点 ---")
print("1. 默认 errorbar=('ci', 95)：95% 置信区间（bootstrap 重抽样）")
print("2. errorbar=('ci', 68)：68% 置信区间，带更窄，约等于均值±1标准误")
print("3. errorbar='sd'：展示±1标准差范围，反映数据离散程度")
print("4. errorbar=None：关闭置信区间带")

# ============================================================
# 5. sns.relplot() — 图级关系图（支持分面）
# ============================================================
# relplot 是图级函数（figure-level），结合了 scatterplot/lineplot 和 FacetGrid
# 通过 kind 参数选择 'scatter'（默认）或 'line'
# 支持 col/row/hue 参数进行分面，适合多组比较

# kind='scatter' + hue 分组
g1 = sns.relplot(data=tips, x="total_bill", y="tip", hue="smoker", kind="scatter", height=5, aspect=1.2)
g1.figure.suptitle("relplot：kind='scatter', hue='smoker'", y=1.02)
g1.figure.savefig(Path(__file__).parent / "relplot_scatter_hue.png", dpi=150, bbox_inches='tight')
plt.show()

# kind='line' + hue 分组
g2 = sns.relplot(data=tips, x="size", y="total_bill", hue="smoker", kind="line", height=5, aspect=1.2)
g2.figure.suptitle("relplot：kind='line', hue='smoker'", y=1.02)
g2.figure.savefig(Path(__file__).parent / "relplot_line_hue.png", dpi=150, bbox_inches='tight')
plt.show()

# col 参数：按列分面
g3 = sns.relplot(
    data=tips, x="total_bill", y="tip",
    col="smoker", kind="scatter", height=5, aspect=1
)
g3.figure.suptitle("relplot：col='smoker' 按列分面", y=1.02)
g3.figure.savefig(Path(__file__).parent / "relplot_col.png", dpi=150, bbox_inches='tight')
plt.show()

# col + row 参数：二维分面
g4 = sns.relplot(
    data=tips, x="total_bill", y="tip",
    col="smoker", row="time", kind="scatter", height=4, aspect=1
)
g4.figure.suptitle("relplot：col='smoker' + row='time' 二维分面", y=1.05)
g4.figure.savefig(Path(__file__).parent / "relplot_col_row.png", dpi=150, bbox_inches='tight')
plt.show()

# col_wrap 参数：当分类水平较多时自动换行
g5 = sns.relplot(
    data=tips, x="total_bill", y="tip",
    col="day", col_wrap=2, kind="scatter", height=4, aspect=1
)
g5.figure.suptitle("relplot：col='day', col_wrap=2 自动换行", y=1.02)
g5.figure.savefig(Path(__file__).parent / "relplot_col_wrap.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- relplot 要点 ---")
print("1. relplot 是图级函数，自动创建 FacetGrid，支持 col/row/hue 分面")
print("2. kind='scatter'（默认）调用 scatterplot，kind='line' 调用 lineplot")
print("3. col: 按列方向分面（水平排列子图）")
print("4. row: 按行方向分面（垂直排列子图）")
print("5. col_wrap: 当分类水平较多时，指定每行子图数，自动换行")
print("6. height + aspect 控制每个子图的大小（height=高度, aspect=宽高比）")

# ============================================================
# 6. scatterplot vs lineplot 使用场景对比
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# scatterplot：展示所有原始数据点
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker", alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title("scatterplot：展示所有原始数据点")
axes[0, 0].set_xlabel("账单总额")
axes[0, 0].set_ylabel("小费")

# lineplot：展示聚合趋势 + 不确定性
sns.lineplot(data=tips, x="total_bill", y="tip", hue="smoker", ax=axes[0, 1])
axes[0, 1].set_title("lineplot：展示聚合趋势 + 置信区间")
axes[0, 1].set_xlabel("账单总额")
axes[0, 1].set_ylabel("小费")

# scatterplot：iris 品种分离
sns.scatterplot(data=iris, x="petal_length", y="petal_width", hue="species", ax=axes[1, 0])
axes[1, 0].set_title("scatterplot：iris 品种分离（无重复x值）")
axes[1, 0].set_xlabel("花瓣长度 (cm)")
axes[1, 0].set_ylabel("花瓣宽度 (cm)")

# lineplot：适合有序/时间序列数据
# 构造模拟时间序列：按用餐人数聚合
tips_agg = tips.groupby("size").agg(mean_bill=("total_bill", "mean"), mean_tip=("tip", "mean")).reset_index()
sns.lineplot(data=tips_agg, x="size", y="mean_bill", marker="o", ax=axes[1, 1], label="账单均值")
sns.lineplot(data=tips_agg, x="size", y="mean_tip", marker="s", ax=axes[1, 1], label="小费均值")
axes[1, 1].set_title("lineplot：有序变量的趋势变化")
axes[1, 1].set_xlabel("用餐人数")
axes[1, 1].set_ylabel("金额")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / "scatter_vs_line.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- scatterplot vs lineplot 使用场景 ---")
print("scatterplot 适合:")
print("  - 展示原始数据点的分布和密度")
print("  - 观察两个变量之间的整体关系模式")
print("  - 数据点较少或 x 值无重复时")
print("  - 需要识别离群值和聚类时")
print("lineplot 适合:")
print("  - 展示趋势变化（时间序列、有序变量）")
print("  - x 值有重复，需要聚合展示均值时")
print("  - 需要展示估计的不确定性（置信区间）时")
print("  - 比较不同组的变化趋势时")

# ============================================================
# 7. 自定义关系图样式
# ============================================================
# markers: 控制标记样式
# palette: 控制配色方案
# alpha: 控制透明度
# sizes: 控制 size 映射的范围
# linewidth: 控制线宽

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# alpha 透明度：解决数据点重叠问题
sns.scatterplot(data=tips, x="total_bill", y="tip", alpha=0.3, ax=axes[0, 0])
axes[0, 0].set_title("alpha=0.3：透明度缓解重叠")
axes[0, 0].set_xlabel("账单总额")
axes[0, 0].set_ylabel("小费")

# palette 配色方案
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", palette="Set2", ax=axes[0, 1])
axes[0, 1].set_title("palette='Set2'：使用 Seaborn 调色板")
axes[0, 0].set_xlabel("账单总额")
axes[0, 0].set_ylabel("小费")

# markers 自定义标记
sns.scatterplot(
    data=tips, x="total_bill", y="tip", hue="smoker", style="smoker",
    markers={"Yes": "D", "No": "o"}, ax=axes[0, 2]
)
axes[0, 2].set_title("自定义 markers：D=菱形, o=圆形")
axes[0, 2].set_xlabel("账单总额")
axes[0, 2].set_ylabel("小费")

# sizes 参数：控制 size 映射的数值范围
sns.scatterplot(
    data=tips, x="total_bill", y="tip", size="size",
    sizes=(20, 200), ax=axes[1, 0]
)
axes[1, 0].set_title("sizes=(20,200)：控制点大小范围")
axes[1, 0].set_xlabel("账单总额")
axes[1, 0].set_ylabel("小费")

# lineplot 自定义：线宽 + 标记 + 配色
sns.lineplot(
    data=tips, x="size", y="total_bill", hue="smoker",
    style="smoker", markers=True, linewidth=2.5,
    palette="muted", ax=axes[1, 1]
)
axes[1, 1].set_title("lineplot 自定义：线宽+标记+配色")
axes[1, 1].set_xlabel("用餐人数")
axes[1, 1].set_ylabel("账单总额")

# 综合自定义：alpha + palette + sizes + markers
sns.scatterplot(
    data=tips, x="total_bill", y="tip",
    hue="day", style="time", size="size",
    palette="deep", sizes=(30, 150), alpha=0.7,
    markers={"Dinner": "s", "Lunch": "^"},
    ax=axes[1, 2]
)
axes[1, 2].set_title("综合自定义：多参数组合")
axes[1, 2].set_xlabel("账单总额")
axes[1, 2].set_ylabel("小费")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "relplot_custom_style.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 关系图自定义样式要点 ---")
print("1. alpha: 透明度（0~1），值越小越透明，适合数据点密集时缓解重叠")
print("2. palette: 配色方案，可选 'deep', 'muted', 'Set2', 'viridis' 等")
print("3. markers: 自定义标记样式，如 {'Yes': 'D', 'No': 'o'}")
print("4. sizes: 元组 (min, max)，控制 size 映射的数值范围")
print("5. linewidth: 线宽（lineplot），默认1，增大可突出趋势线")
print("6. 常用标记: 'o'(圆), 's'(方), 'D'(菱), '^'(三角), '*(星), 'X'(叉)")

# ============================================================
# 8. relplot 综合案例：iris 多维度探索
# ============================================================
# 使用 relplot 对 iris 数据集进行多维度探索
# 同时利用 hue + size + col 展示4个维度的信息

g6 = sns.relplot(
    data=iris, x="sepal_length", y="petal_length",
    hue="species", size="petal_width",
    sizes=(20, 200), alpha=0.7,
    height=6, aspect=1.3,
    palette="Set1"
)
g6.figure.suptitle("iris 多维度关系图：hue=品种, size=花瓣宽度", y=1.02)
g6.figure.savefig(Path(__file__).parent / "relplot_iris_multi.png", dpi=150, bbox_inches='tight')
plt.show()

# 按品种分面
g7 = sns.relplot(
    data=iris, x="sepal_length", y="petal_length",
    hue="petal_width", size="petal_width",
    col="species", col_wrap=2,
    sizes=(20, 150), alpha=0.7,
    height=4, aspect=1,
    palette="YlOrRd"
)
g7.figure.suptitle("iris 按品种分面：颜色和大小映射花瓣宽度", y=1.03)
g7.figure.savefig(Path(__file__).parent / "relplot_iris_facet.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- iris 多维度探索要点 ---")
print("1. hue + size 可同时编码两个数值维度，信息密度高")
print("2. col 分面将不同品种分开观察，避免颜色混淆")
print("3. palette='YlOrRd' 等连续色板适合映射连续数值变量")
print("4. Setosa 品种花瓣最短最窄，Virginica 花瓣最长最宽")

# ============================================================
# 9. scatterplot 与 lineplot 轴级 vs relplot 图级对比
# ============================================================
print("\n--- 轴级函数 vs 图级函数 ---")
print("scatterplot / lineplot（轴级 axes-level）:")
print("  - 可指定 ax 参数，灵活嵌入 matplotlib 自定义布局")
print("  - 适合在同一个 figure 中组合多种图表")
print("  - 返回 Axes 对象")
print("relplot（图级 figure-level）:")
print("  - 自动创建 FacetGrid，支持 col/row 分面")
print("  - 适合快速生成多子图比较")
print("  - 返回 FacetGrid 对象，通过 .figure 访问 Figure")
print("  - 不能指定 ax 参数，布局由 FacetGrid 控制")
print("选择建议:")
print("  - 需要自定义布局或组合图表 → scatterplot / lineplot")
print("  - 需要分面比较多组数据 → relplot")
print("  - 两者底层绘图逻辑相同，只是接口层级不同")
