# 数据来源: seaborn 内置数据集 tips, penguins
# 分面网格（FacetGrid）是 Seaborn 中最强大的多面板绘图工具
# 本案例覆盖: FacetGrid, map, map_dataframe, col/row/hue, add_legend, set_axis_labels, set_titles, catplot
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins").dropna()
print("tips 数据集前5行:")
print(tips.head())
print("\npenguins 数据集前5行:")
print(penguins.head())

# ============================================================
# 1. sns.FacetGrid() 基础用法
# ============================================================
# FacetGrid 创建多面板网格，每个面板展示数据的子集
# 核心步骤：创建网格 → 映射绘图函数 → 添加图例/标签

# 按 time 分面（列方向）
g = sns.FacetGrid(tips, col="time", height=5, aspect=1)
g.map(sns.histplot, "total_bill")
g.figure.suptitle("FacetGrid：按用餐时间分面的账单分布", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_col.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- FacetGrid 基础 ---")
print("1. FacetGrid(data, col=...) 按列方向分面")
print("2. .map(func, 'col_name') 将绘图函数映射到每个面板")
print("3. .map() 传递的函数必须是轴级函数（如 histplot, scatterplot）")

# ============================================================
# 2. col + row 二维分面
# ============================================================
g = sns.FacetGrid(tips, col="time", row="smoker", height=4, aspect=1)
g.map(sns.histplot, "total_bill", bins=15)
g.figure.suptitle("FacetGrid：time(列) × smoker(行) 二维分面", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_col_row.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- col + row 分面 ---")
print("col: 列方向分面（水平排列）")
print("row: 行方向分面（垂直排列）")
print("col + row: 二维网格，每个子集一个面板")

# ============================================================
# 3. hue 参数 + add_legend()
# ============================================================
# hue 按颜色区分组别，不创建新面板，而是在同一面板内叠加

g = sns.FacetGrid(tips, col="time", hue="smoker", height=5, aspect=1)
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.add_legend()
g.figure.suptitle("FacetGrid：hue='smoker' 颜色分组 + 图例", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_hue.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- hue 参数 ---")
print("hue: 在同一面板内按颜色区分组别")
print("add_legend(): 添加颜色图例")
print("hue 不创建新面板，而是叠加在同一面板中")

# ============================================================
# 4. .map() vs .map_dataframe()
# ============================================================
# .map(): 直接传递列名，函数接收位置参数
# .map_dataframe(): 传递整个DataFrame，函数接收关键字参数 data=

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# .map() 用法：传递列名作为位置参数
g1 = sns.FacetGrid(tips, col="time", height=5, aspect=1)
g1.map(plt.scatter, "total_bill", "tip", alpha=0.5, edgecolor="w")
g1.figure.suptitle(".map()：传递列名作为位置参数", y=1.02)

# .map_dataframe() 用法：可以传递 data= 参数
g2 = sns.FacetGrid(tips, col="time", hue="smoker", height=5, aspect=1)
g2.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.7)
g2.add_legend()
g2.figure.suptitle(".map_dataframe()：支持关键字参数", y=1.02)

plt.show()

print("\n--- .map() vs .map_dataframe() ---")
print(".map(func, 'x', 'y'): 列名作为位置参数传递")
print(".map_dataframe(func, x='x', y='y'): 支持关键字参数，更灵活")
print("Seaborn 自带函数推荐用 .map_dataframe()，可以传递 hue 等参数")

# ============================================================
# 5. .set_axis_labels() 与 .set_titles()
# ============================================================
g = sns.FacetGrid(tips, col="time", row="smoker", hue="sex", height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.7)
g.add_legend()
g.set_axis_labels("账单总额（美元）", "小费（美元）")
g.set_titles(col_template="{col_name}用餐", row_template="吸烟: {row_name}")
g.figure.suptitle("自定义轴标签和面板标题", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_labels.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 标签与标题自定义 ---")
print("set_axis_labels(x_label, y_label): 设置所有面板的轴标签")
print("set_titles(col_template, row_template): 自定义面板标题模板")
print("模板变量: {col_name}, {row_name}, {var_name} 等")

# ============================================================
# 6. FacetGrid + 回归图
# ============================================================
g = sns.FacetGrid(tips, col="time", row="smoker", height=4, aspect=1)
g.map_dataframe(sns.regplot, x="total_bill", y="tip",
                scatter_kws={"alpha": 0.5, "s": 20},
                line_kws={"color": "red"})
g.set_axis_labels("账单总额", "小费")
g.set_titles(col_template="{col_name}", row_template="吸烟: {row_name}")
g.figure.suptitle("FacetGrid + regplot：分组回归分析", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_regplot.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. FacetGrid + 多种图表类型
# ============================================================
# 使用 penguins 数据集展示更丰富的分面可视化

g = sns.FacetGrid(penguins, col="species", row="island", hue="sex", height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x="flipper_length_mm", y="body_mass_g", alpha=0.7)
g.add_legend()
g.set_axis_labels("鳍长 (mm)", "体重 (g)")
g.set_titles(col_template="{col_name}", row_template="{row_name}岛")
g.figure.suptitle("企鹅数据：物种 × 岛屿 分面散点图", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_penguins.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. sns.catplot() — 分类图的快捷方式
# ============================================================
# catplot 是图级函数，内部使用 FacetGrid，kind 参数选择图表类型
# kind: "strip"(默认), "swarm", "box", "violin", "boxen", "point", "bar", "count"

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

kinds = ["strip", "swarm", "box", "violin", "bar", "point"]
for idx, kind in enumerate(kinds):
    row, col = divmod(idx, 3)
    g = sns.catplot(data=tips, x="day", y="total_bill", kind=kind, height=4, aspect=1)
    g.figure.suptitle(f"catplot kind='{kind}'", y=1.02)
    plt.close(g.figure)

    ax = axes[row][col]
    sns_func_map = {
        "strip": sns.stripplot, "swarm": sns.swarmplot,
        "box": sns.boxplot, "violin": sns.violinplot,
        "bar": sns.barplot, "point": sns.pointplot
    }
    sns_func_map[kind](data=tips, x="day", y="total_bill", ax=ax)
    ax.set_title(f"kind='{kind}'")
    ax.set_xlabel("星期")
    ax.set_ylabel("账单总额")

plt.suptitle("catplot 六种 kind 类型对比", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "catplot_kinds.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- catplot 要点 ---")
print("catplot 是图级函数，自动创建 FacetGrid")
print("kind 参数选择图表类型: strip, swarm, box, violin, boxen, point, bar, count")
print("支持 col/row/hue 参数进行分面")
print("catplot(kind='box', col='time') 等价于 FacetGrid + map(boxplot)")

# catplot 分面示例
g = sns.catplot(
    data=tips, x="day", y="total_bill",
    kind="box", col="time", row="smoker",
    height=4, aspect=1
)
g.figure.suptitle("catplot：kind='box' + col/row 分面", y=1.02)
g.figure.savefig(Path(__file__).parent / "catplot_facet.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. FacetGrid 高级自定义
# ============================================================
g = sns.FacetGrid(tips, col="day", height=4, aspect=1, col_wrap=2)
g.map_dataframe(sns.kdeplot, x="total_bill", fill=True, alpha=0.5)
g.set_axis_labels("账单总额", "密度")
g.set_titles("{col_name}")
g.figure.suptitle("FacetGrid + kdeplot：每日账单核密度估计", y=1.02)
g.figure.savefig(Path(__file__).parent / "facetgrid_kde.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- FacetGrid 高级用法 ---")
print("col_wrap: 当列数过多时自动换行（如 col_wrap=2 每行2个面板）")
print("height + aspect: 控制每个面板的尺寸（width = height × aspect）")
print("despine: g.fig.subplots_adjust() 可调整面板间距")
