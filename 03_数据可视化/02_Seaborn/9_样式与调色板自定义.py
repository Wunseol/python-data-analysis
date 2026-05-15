# 数据来源: seaborn 内置数据集 tips, penguins
# Seaborn 提供了丰富的样式和调色板自定义功能
# 本案例覆盖: set_style, set_context, set_palette, color_palette, despine, 自定义调色板, rcParams集成
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

tips = sns.load_dataset("tips")
penguins = sns.load_dataset("penguins").dropna()

# ============================================================
# 1. sns.set_style() — 预设样式
# ============================================================
# Seaborn 提供5种预设样式: darkgrid, whitegrid, dark, white, ticks
# darkgrid(默认): 深色背景+网格线，适合数据探索
# whitegrid: 白色背景+网格线，适合正式报告
# dark: 深色背景无网格
# white: 白色背景无网格
# ticks: 白色背景+刻度线

styles = ["darkgrid", "whitegrid", "dark", "white", "ticks"]

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for idx, style in enumerate(styles):
    with sns.axes_style(style):
        ax = axes[idx]
        sns.histplot(data=tips, x="total_bill", ax=ax)
        ax.set_title(f"style='{style}'")
        ax.set_xlabel("账单总额")
        ax.set_ylabel("频数")

plt.suptitle("Seaborn 五种预设样式对比", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "set_style_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("--- set_style 要点 ---")
print("darkgrid: 深色背景+网格（默认，适合数据探索）")
print("whitegrid: 白色背景+网格（适合正式报告）")
print("dark: 深色背景无网格")
print("white: 白色背景无网格")
print("ticks: 白色背景+刻度线（学术论文常用）")
print("提示: 使用 with sns.axes_style() 临时切换样式，不影响全局")

# ============================================================
# 2. sns.set_context() — 上下文缩放
# ============================================================
# set_context 控制图表元素的大小比例，适应不同使用场景
# 四种预设: paper, notebook, talk, poster
# paper: 最小（论文用）
# notebook: 中等（默认，笔记本用）
# talk: 较大（演讲用）
# poster: 最大（海报用）

contexts = ["paper", "notebook", "talk", "poster"]

fig, axes = plt.subplots(1, 4, figsize=(24, 5))
for idx, context in enumerate(contexts):
    with sns.plotting_context(context):
        ax = axes[idx]
        sns.barplot(data=tips, x="day", y="total_bill", ax=ax)
        ax.set_title(f"context='{context}'")
        ax.set_xlabel("星期")
        ax.set_ylabel("账单总额")

plt.suptitle("Seaborn 四种上下文缩放对比", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "set_context_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- set_context 要点 ---")
print("paper: 元素最小，适合论文插图")
print("notebook: 元素中等（默认），适合Jupyter笔记本")
print("talk: 元素较大，适合演讲PPT")
print("poster: 元素最大，适合学术海报")
print("可微调: set_context('talk', rc={'font.size': 14})")

# ============================================================
# 3. sns.set_palette() 与 sns.color_palette() — 调色板
# ============================================================
# set_palette: 全局设置调色板
# color_palette: 返回调色板颜色列表（不修改全局设置）

# Seaborn 6种定性调色板
palette_types = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]

fig, axes = plt.subplots(6, 1, figsize=(10, 12))
for idx, palette in enumerate(palette_types):
    pal = sns.color_palette(palette)
    sns.palplot(pal, ax=axes[idx])
    axes[idx].set_title(f"palette='{palette}'", loc='left')

plt.suptitle("Seaborn 六种定性调色板", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "palette_types.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 定性调色板 ---")
print("deep:     默认调色板，色彩鲜明，区分度好")
print("muted:    柔和色调，适合多组比较")
print("bright:   高饱和度，适合投影展示")
print("pastel:   粉彩色调，适合温和展示")
print("dark:     深色调，适合浅色背景")
print("colorblind: 色盲友好，红绿色盲可区分")

# 使用不同调色板绘图对比
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, palette in enumerate(palette_types):
    row, col = divmod(idx, 3)
    sns.set_palette(palette)
    sns.barplot(data=tips, x="day", y="total_bill", hue="sex", ax=axes[row][col])
    axes[row][col].set_title(f"palette='{palette}'")
    axes[row][col].set_xlabel("星期")
    axes[row][col].set_ylabel("账单总额")

sns.set_palette("deep")
plt.suptitle("不同调色板在条形图中的效果", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "palette_barplot.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. 序列调色板与发散调色板
# ============================================================
# 序列调色板: 适合连续数值（如热力图）
# 发散调色板: 适合有中心值的数据（如相关系数）

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 序列调色板
seq_palettes = ["Blues", "Greens", "YlOrRd"]
for idx, pal in enumerate(seq_palettes):
    pal_colors = sns.color_palette(pal, as_cmap=True)
    sns.heatmap(
        tips[["total_bill", "tip", "size"]].corr(),
        annot=True, cmap=pal_colors, ax=axes[0][idx]
    )
    axes[0][idx].set_title(f"序列调色板: {pal}")

# 发散调色板
div_palettes = ["RdBu", "coolwarm", "vlag"]
for idx, pal in enumerate(div_palettes):
    pal_colors = sns.color_palette(pal, as_cmap=True)
    sns.heatmap(
        tips[["total_bill", "tip", "size"]].corr(),
        annot=True, cmap=pal_colors, center=0, ax=axes[1][idx]
    )
    axes[1][idx].set_title(f"发散调色板: {pal}")

plt.suptitle("序列调色板 vs 发散调色板", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sequential_diverging_palettes.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 序列与发散调色板 ---")
print("序列调色板: 单色渐变，适合连续数值（如 Blues, Greens, YlOrRd）")
print("发散调色板: 双色渐变，适合有中心值的数据（如 RdBu, coolwarm, vlag）")
print("使用 as_cmap=True 返回 matplotlib Colormap 对象")

# ============================================================
# 5. 自定义调色板
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 方式1: 传入颜色列表
custom_list = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
sns.barplot(data=tips, x="day", y="total_bill", palette=custom_list, ax=axes[0][0])
axes[0][0].set_title("自定义颜色列表（HEX）")

# 方式2: 使用 sns.color_palette() 指定颜色数量
pal_8 = sns.color_palette("husl", 8)
sns.barplot(data=tips, x="day", y="total_bill", palette=pal_8, ax=axes[0][1])
axes[0][1].set_title("husl 调色板（8色）")

# 方式3: 使用 light_palette / dark_palette 生成
light_pal = sns.light_palette("seagreen", as_cmap=True)
sns.heatmap(tips[["total_bill", "tip", "size"]].corr(), annot=True, cmap=light_pal, ax=axes[0][2])
axes[0][2].set_title("light_palette('seagreen')")

# 方式4: 使用 diverging_palette 生成发散调色板
div_pal = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(tips[["total_bill", "tip", "size"]].corr(), annot=True, cmap=div_pal, center=0, ax=axes[1][0])
axes[1][0].set_title("diverging_palette(220, 20)")

# 方式5: 使用 cubehelix_palette
cube_pal = sns.cubehelix_palette(as_cmap=True)
sns.heatmap(tips[["total_bill", "tip", "size"]].corr(), annot=True, cmap=cube_pal, ax=axes[1][1])
axes[1][1].set_title("cubehelix_palette()")

# 方式6: 使用 matplotlib Colormap
mpl_cmap = plt.cm.plasma
sns.heatmap(tips[["total_bill", "tip", "size"]].corr(), annot=True, cmap=mpl_cmap, ax=axes[1][2])
axes[1][2].set_title("matplotlib Colormap: plasma")

plt.suptitle("自定义调色板六种方式", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "custom_palettes.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- 自定义调色板方式 ---")
print("1. 颜色列表: ['#FF6B6B', '#4ECDC4', ...]（HEX格式）")
print("2. color_palette('husl', n): 均匀分布的HSL颜色")
print("3. light_palette / dark_palette: 从基色生成渐变")
print("4. diverging_palette(h_neg, h_pos): 生成发散调色板")
print("5. cubehelix_palette: 灰度友好的渐变调色板")
print("6. matplotlib Colormap: plt.cm.xxx 系列")

# ============================================================
# 6. sns.despine() — 移除边框
# ============================================================
# despine 移除图表的上边框和右边框，使图表更简洁

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 默认边框
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[0])
axes[0].set_title("默认边框")

# despine() 移除上和右边框
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[1])
sns.despine(ax=axes[1])
axes[1].set_title("despine(): 移除上和右边框")

# despine 移除所有边框
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[2])
sns.despine(ax=axes[2], left=True, bottom=True, right=True, top=True)
axes[2].set_title("despine(all): 移除所有边框")

plt.suptitle("sns.despine() 边框控制", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "despine_demo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- despine 要点 ---")
print("despine(): 默认移除 top 和 right 边框")
print("despine(left=True, bottom=True): 移除所有边框")
print("despine(offset=10): 边框与轴偏移10个点")
print("despine(trim=True): 修剪边框至数据范围")

# ============================================================
# 7. plt.rcParams 集成
# ============================================================
# Seaborn 的样式系统与 matplotlib 的 rcParams 完全兼容
# 可以在 Seaborn 样式基础上微调 matplotlib 参数

# 先设置 Seaborn 样式
sns.set_style("whitegrid")
sns.set_context("talk")

# 再用 rcParams 微调
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#cccccc',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Arial'],
    'axes.unicode_minus': False,
    'figure.dpi': 100,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=tips, x="day", y="total_bill", hue="sex", ax=axes[0])
axes[0].set_title("Seaborn + rcParams 自定义样式")
axes[0].set_xlabel("星期")
axes[0].set_ylabel("账单总额")

sns.violinplot(data=penguins, x="species", y="body_mass_g", hue="sex", split=True, ax=axes[1])
axes[1].set_title("Seaborn + rcParams 自定义样式")
axes[1].set_xlabel("物种")
axes[1].set_ylabel("体重 (g)")

sns.despine()
plt.tight_layout()
plt.savefig(Path(__file__).parent / "rcparams_integration.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n--- rcParams 集成要点 ---")
print("1. sns.set_style() 底层修改的是 plt.rcParams")
print("2. 可以在 set_style 之后用 plt.rcParams.update() 微调")
print("3. 常用参数: figure.facecolor, axes.facecolor, font.size 等")
print("4. sns.set_theme() = set_style() + set_context() + set_palette() 一次性设置")

# ============================================================
# 8. 样式组合实战：学术论文风格
# ============================================================
# 恢复默认设置
sns.reset_defaults()
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

with sns.axes_style("ticks"):
    with sns.plotting_context("paper", rc={"font.size": 12, "axes.labelsize": 14}):
        g = sns.FacetGrid(tips, col="time", hue="smoker", height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.7, s=30)
        g.add_legend()
        g.set_axis_labels("账单总额（美元）", "小费（美元）")
        g.set_titles("{col_name}")
        g.figure.suptitle("学术论文风格：ticks + paper context", y=1.02)
        sns.despine(trim=True)
        g.figure.savefig(Path(__file__).parent / "academic_style.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n--- 学术论文风格建议 ---")
print("样式: 'ticks' 或 'white'（简洁无网格）")
print("上下文: 'paper'（元素较小）")
print("分辨率: dpi=300 以上")
print("格式: 保存为 PDF 或 SVG（矢量图）")
print("despine(trim=True): 修剪边框至数据范围")
