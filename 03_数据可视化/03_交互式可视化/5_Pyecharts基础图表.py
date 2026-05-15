# 数据来源: 自造模拟数据
# Pyecharts 是基于 ECharts 的 Python 可视化库，生成的图表交互性强
# 特点: 链式调用配置、丰富的图表类型、中国地图支持好、可导出HTML
# 依赖: pip install pyecharts

from pyecharts.charts import Bar, Line, Pie, Scatter
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import random

print("=" * 60)
print("Pyecharts 基础图表")
print("=" * 60)

# ============================================================
# 1. 柱状图 - Bar()
# ============================================================
print("\n--- 1. 柱状图 Bar() ---")

months = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]
sales_a = [random.randint(100, 500) for _ in range(12)]
sales_b = [random.randint(80, 400) for _ in range(12)]

bar = (
    Bar(init_opts=opts.InitOpts(
        width="900px",
        height="500px",
        theme=ThemeType.LIGHT,
    ))
    .add_xaxis(months)
    .add_yaxis("产品A", sales_a, color="#5470C6")
    .add_yaxis("产品B", sales_b, color="#91CC75")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="月度产品销售额对比", subtitle="模拟数据"),
        xaxis_opts=opts.AxisOpts(name="月份"),
        yaxis_opts=opts.AxisOpts(name="销售额 (万元)"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
    )
)

bar.render("output_pyecharts_柱状图.html")
print("柱状图已保存为 output_pyecharts_柱状图.html")

# ============================================================
# 2. 折线图 - Line()
# ============================================================
print("\n--- 2. 折线图 Line() ---")

temp_beijing = [random.randint(-5, 35) for _ in range(12)]
temp_shanghai = [random.randint(0, 38) for _ in range(12)]
temp_guangzhou = [random.randint(5, 38) for _ in range(12)]

line = (
    Line(init_opts=opts.InitOpts(
        width="900px",
        height="500px",
        theme=ThemeType.MACARONS,
    ))
    .add_xaxis(months)
    .add_yaxis(
        "北京",
        temp_beijing,
        is_smooth=True,
        linestyle_opts=opts.LineStyleOpts(width=3),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        "上海",
        temp_shanghai,
        is_smooth=True,
        linestyle_opts=opts.LineStyleOpts(width=3),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_yaxis(
        "广州",
        temp_guangzhou,
        is_smooth=True,
        linestyle_opts=opts.LineStyleOpts(width=3),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="三城市月度气温变化", subtitle="模拟数据"),
        xaxis_opts=opts.AxisOpts(name="月份"),
        yaxis_opts=opts.AxisOpts(name="温度 (°C)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
    )
)

line.render("output_pyecharts_折线图.html")
print("折线图已保存为 output_pyecharts_折线图.html")

# ============================================================
# 3. 饼图 - Pie()
# ============================================================
print("\n--- 3. 饼图 Pie() ---")

market_data = [
    ("Chrome", 65),
    ("Safari", 18),
    ("Firefox", 8),
    ("Edge", 5),
    ("其他", 4),
]

pie = (
    Pie(init_opts=opts.InitOpts(
        width="800px",
        height="600px",
        theme=ThemeType.ROMANTIC,
    ))
    .add(
        "浏览器",
        market_data,
        radius=["30%", "70%"],
        rosetype="radius",
        label_opts=opts.LabelOpts(formatter="{b}: {d}%"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="浏览器市场份额", subtitle="模拟数据"),
        legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
    )
)

pie.render("output_pyecharts_饼图.html")
print("饼图已保存为 output_pyecharts_饼图.html")

# ============================================================
# 4. 散点图 - Scatter()
# ============================================================
print("\n--- 4. 散点图 Scatter() ---")

random.seed(42)
x_data = [random.uniform(0, 100) for _ in range(50)]
y_data = [0.8 * x + random.uniform(-20, 20) for x in x_data]

scatter = (
    Scatter(init_opts=opts.InitOpts(
        width="800px",
        height="600px",
        theme=ThemeType.WONDERLAND,
    ))
    .add_xaxis([round(x, 1) for x in x_data])
    .add_yaxis(
        "数据点",
        [round(y, 1) for y in y_data],
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="散点图: 变量相关性分析", subtitle="模拟数据"),
        xaxis_opts=opts.AxisOpts(name="变量X", type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)),
        yaxis_opts=opts.AxisOpts(name="变量Y", type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)),
        tooltip_opts=opts.TooltipOpts(formatter="{b}: ({c})"),
        visualmap_opts=opts.VisualMapOpts(
            type_="size",
            max_=120,
            min_=0,
            range_size=[5, 20],
        ),
    )
)

scatter.render("output_pyecharts_散点图.html")
print("散点图已保存为 output_pyecharts_散点图.html")

# ============================================================
# 5. 链式配置详解
# ============================================================
print("\n--- 5. 链式配置说明 ---")
print("""
Pyecharts 的链式调用风格:
    chart = (
        Bar()
        .add_xaxis([...])        # 添加X轴数据
        .add_yaxis("系列1", [...]) # 添加Y轴数据系列
        .add_yaxis("系列2", [...]) # 可添加多个系列
        .set_global_opts(...)     # 全局配置 (标题/坐标轴/图例等)
        .set_series_opts(...)     # 系列配置 (标签/标记等)
    )
    chart.render("output.html")   # 渲染为HTML文件

常用全局配置 set_global_opts:
    - title_opts: 标题配置
    - xaxis_opts / yaxis_opts: 坐标轴配置
    - legend_opts: 图例配置
    - tooltip_opts: 提示框配置
    - visualmap_opts: 视觉映射配置
    - datazoom_opts: 数据缩放配置

常用系列配置 set_series_opts:
    - label_opts: 标签配置
    - markpoint_opts: 标记点配置
    - markline_opts: 标记线配置
    - linestyle_opts: 线样式配置
    - areastyle_opts: 区域填充样式
""")

# ============================================================
# 6. 组合图表 - 柱状图+折线图
# ============================================================
print("\n--- 6. 组合图表: 柱状图+折线图 ---")

bar_line = (
    Bar(init_opts=opts.InitOpts(
        width="900px",
        height="500px",
        theme=ThemeType.ESSOS,
    ))
    .add_xaxis(months)
    .add_yaxis(
        "销售额",
        sales_a,
        color="#5470C6",
        yaxis_index=0,
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="增长率 (%)",
            type_="value",
            position="right",
            axislabel_opts=opts.LabelOpts(formatter="{value}%"),
        )
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="销售额与增长率组合图", subtitle="模拟数据"),
        yaxis_opts=opts.AxisOpts(name="销售额 (万元)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
    )
)

growth_rate = [round(random.uniform(-10, 30), 1) for _ in range(12)]
line_overlay = (
    Line()
    .add_xaxis(months)
    .add_yaxis(
        "增长率",
        growth_rate,
        yaxis_index=1,
        linestyle_opts=opts.LineStyleOpts(width=3, color="#EE6666"),
        label_opts=opts.LabelOpts(formatter="{c}%"),
    )
)

bar_line.overlap(line_overlay)
bar_line.render("output_pyecharts_组合图.html")
print("组合图表已保存为 output_pyecharts_组合图.html")

print("\n" + "=" * 60)
print("Pyecharts 基础图表演示完成！")
print("所有HTML文件已保存到当前目录，可在浏览器中打开查看交互效果。")
print("=" * 60)
