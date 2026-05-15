# 数据来源: 自造模拟数据
# Bokeh 是一个专注于浏览器交互式可视化的Python库
# 特点: 高性能大数据渲染、丰富的交互工具、可扩展的Widget系统
# 依赖: pip install bokeh

from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS
from bokeh.layouts import column, row, gridplot
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Category10, Viridis256
import numpy as np
import pandas as pd

print("=" * 60)
print("Bokeh 交互式可视化")
print("=" * 60)

# ============================================================
# 1. 基础图表 - figure(), circle(), line()
# ============================================================
print("\n--- 1. 基础图表 ---")

output_file("output_bokeh_基础图表.html")

np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 0.8 * x + np.random.randn(n) * 0.5

p1 = figure(
    title="散点图: 变量相关性",
    width=600,
    height=400,
    tools="pan,box_zoom,wheel_zoom,reset,save",
)
p1.circle(x, y, size=8, color="navy", alpha=0.6)
p1.xaxis.axis_label = "变量X"
p1.yaxis.axis_label = "变量Y"

x_line = np.linspace(0, 10, 50)
y_line = np.sin(x_line) + np.random.randn(50) * 0.2

p2 = figure(
    title="折线图: 带噪声的正弦波",
    width=600,
    height=400,
    tools="pan,box_zoom,wheel_zoom,reset,save",
)
p2.line(x_line, y_line, line_width=2, color="firebrick", legend_label="sin(x)+noise")
p2.circle(x_line, y_line, size=4, color="black", alpha=0.5)
p2.xaxis.axis_label = "X"
p2.yaxis.axis_label = "Y"

grid = gridplot([[p1, p2]])
save(grid)
print("基础图表已保存为 output_bokeh_基础图表.html")

# ============================================================
# 2. ColumnDataSource - 数据源
# ============================================================
print("\n--- 2. ColumnDataSource 数据源 ---")

output_file("output_bokeh_数据源.html")

df_fruits = pd.DataFrame({
    "水果": ["苹果", "香蕉", "橙子", "葡萄", "草莓"],
    "销量": [120, 85, 95, 60, 110],
    "价格": [5.5, 3.2, 4.8, 12.0, 15.5],
    "颜色": Category10[5],
})

source = ColumnDataSource(df_fruits)

p_bar = figure(
    x_range=df_fruits["水果"].tolist(),
    title="水果销量柱状图",
    width=700,
    height=400,
    tools="pan,wheel_zoom,reset,save",
)

p_bar.vbar(
    x="水果",
    top="销量",
    width=0.6,
    source=source,
    fill_color=factor_cmap("水果", palette=Category10[5], factors=df_fruits["水果"].tolist()),
    line_color="white",
)

p_bar.xaxis.axis_label = "水果"
p_bar.yaxis.axis_label = "销量 (斤)"

save(p_bar)
print("ColumnDataSource图表已保存为 output_bokeh_数据源.html")

# ============================================================
# 3. HoverTool - 悬停提示
# ============================================================
print("\n--- 3. HoverTool 悬停提示 ---")

output_file("output_bokeh_悬停提示.html")

np.random.seed(42)
n_pts = 200
df_scatter = pd.DataFrame({
    "x": np.random.randn(n_pts) * 10 + 50,
    "y": np.random.randn(n_pts) * 10 + 50,
    "size": np.random.randint(5, 30, n_pts),
    "category": np.random.choice(["A类", "B类", "C类"], n_pts),
    "value": np.random.randint(10, 100, n_pts),
})

source_hover = ColumnDataSource(df_scatter)

hover = HoverTool(
    tooltips=[
        ("类别", "@category"),
        ("坐标X", "@x{0.1f}"),
        ("坐标Y", "@y{0.1f}"),
        ("数值", "@value"),
        ("大小", "@size"),
    ],
)

p_hover = figure(
    title="悬停提示交互散点图",
    width=700,
    height=500,
    tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"],
)

p_hover.scatter(
    x="x",
    y="y",
    size="size",
    color=factor_cmap("category", palette=Category10[3], factors=["A类", "B类", "C类"]),
    alpha=0.7,
    source=source_hover,
    legend_field="category",
)

p_hover.xaxis.axis_label = "X坐标"
p_hover.yaxis.axis_label = "Y坐标"
p_hover.legend.location = "top_right"

save(p_hover)
print("悬停提示图表已保存为 output_bokeh_悬停提示.html")

# ============================================================
# 4. 颜色映射 - linear_cmap
# ============================================================
print("\n--- 4. 颜色映射 ---")

output_file("output_bokeh_颜色映射.html")

np.random.seed(42)
n_cities = 30
df_cities = pd.DataFrame({
    "city": [f"城市{i}" for i in range(n_cities)],
    "longitude": np.random.uniform(100, 130, n_cities),
    "latitude": np.random.uniform(20, 50, n_cities),
    "population": np.random.randint(100, 5000, n_cities),
})

source_cities = ColumnDataSource(df_cities)

hover_city = HoverTool(tooltips=[("城市", "@city"), ("人口", "@population 万")])

p_cities = figure(
    title="城市人口分布 (颜色映射)",
    width=700,
    height=500,
    tools=[hover_city, "pan,wheel_zoom,reset,save"],
)

p_cities.scatter(
    x="longitude",
    y="latitude",
    size=12,
    color=linear_cmap("population", Viridis256, low=100, high=5000),
    alpha=0.8,
    source=source_cities,
)

p_cities.xaxis.axis_label = "经度"
p_cities.yaxis.axis_label = "纬度"

from bokeh.models import ColorBar
from bokeh.transform import linear_cmap as lc

color_mapper = lc("population", Viridis256, low=100, high=5000)
color_bar = ColorBar(color_mapper=color_mapper["transform"], width=8, location=(0, 0), title="人口(万)")
p_cities.add_layout(color_bar, "right")

save(p_cities)
print("颜色映射图表已保存为 output_bokeh_颜色映射.html")

# ============================================================
# 5. 布局系统 - column, row, gridplot
# ============================================================
print("\n--- 5. 布局系统 ---")

output_file("output_bokeh_布局系统.html")

p_a = figure(title="图表A", width=350, height=300)
p_a.circle([1, 2, 3, 4, 5], [2, 5, 3, 4, 1], size=10, color="red")

p_b = figure(title="图表B", width=350, height=300)
p_b.line([1, 2, 3, 4, 5], [2, 5, 3, 4, 1], line_width=2, color="blue")

p_c = figure(title="图表C", width=350, height=300)
p_c.vbar(x=[1, 2, 3, 4, 5], top=[2, 5, 3, 4, 1], width=0.5, color="green")

p_d = figure(title="图表D", width=350, height=300)
p_d.square([1, 2, 3, 4, 5], [2, 5, 3, 4, 1], size=12, color="purple")

layout = gridplot([[p_a, p_b], [p_c, p_d]])

save(layout)
print("布局系统图表已保存为 output_bokeh_布局系统.html")

# ============================================================
# 6. Widget - Slider 滑块交互
# ============================================================
print("\n--- 6. Widget Slider 交互 ---")

output_file("output_bokeh_滑块交互.html")

x_vals = np.linspace(0, 4 * np.pi, 200)
y_vals = np.sin(x_vals)

source_slider = ColumnDataSource(data=dict(x=x_vals, y=y_vals))

p_slider = figure(
    title="正弦波: 调节频率",
    width=700,
    height=400,
    tools="pan,wheel_zoom,reset,save",
)
p_slider.line("x", "y", source=source_slider, line_width=3, color="steelblue")

slider = Slider(start=0.5, end=5.0, value=1.0, step=0.1, title="频率")

callback = CustomJS(
    args=dict(source=source_slider, slider=slider),
    code="""
    var data = source.data;
    var f = slider.value;
    var x = data['x'];
    var y = data['y'];
    for (var i = 0; i < x.length; i++) {
        y[i] = Math.sin(f * x[i]);
    }
    source.change.emit();
    """,
)

slider.js_on_change("value", callback)

layout_slider = column(slider, p_slider)
save(layout_slider)
print("滑块交互图表已保存为 output_bokeh_滑块交互.html")

# ============================================================
# 7. Bokeh 核心概念总结
# ============================================================
print("\n--- 7. Bokeh 核心概念总结 ---")
print("""
Bokeh 核心概念:
1. figure()       - 创建画布，所有图表的容器
2. glyph方法      - circle/line/vbar/square 等绘图原语
3. ColumnDataSource - 统一的数据源对象，支持列名引用
4. HoverTool      - 悬停提示工具，自定义显示内容
5. output_file()  - 指定输出HTML文件
6. show() / save() - 显示或保存图表
7. layout系统     - column/row/gridplot 组织多图表
8. Widget         - Slider/Button/Select 等交互控件
9. CustomJS       - JavaScript回调实现客户端交互
10. 颜色映射      - factor_cmap (分类) / linear_cmap (连续)
""")

print("\n" + "=" * 60)
print("Bokeh 交互式可视化演示完成！")
print("所有HTML文件已保存到当前目录，可在浏览器中打开查看交互效果。")
print("=" * 60)
