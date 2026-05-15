# 数据来源: 自造模拟数据
# Pyecharts 的地图和热力图功能是其特色之一
# 特别适合中国地图可视化，内置省份/城市地理编码
# 依赖: pip install pyecharts (地图数据已内置)

from pyecharts.charts import Map, Geo, HeatMap
from pyecharts import options as opts
from pyecharts.globals import ThemeType, ChartType
import random

print("=" * 60)
print("Pyecharts 地图与热力图")
print("=" * 60)

# ============================================================
# 1. 中国省级地图 - Map()
# ============================================================
print("\n--- 1. 中国省级地图 Map() ---")

province_data = [
    ("北京市", random.randint(5000, 40000)),
    ("天津市", random.randint(3000, 20000)),
    ("河北省", random.randint(5000, 30000)),
    ("山西省", random.randint(3000, 20000)),
    ("内蒙古自治区", random.randint(2000, 15000)),
    ("辽宁省", random.randint(4000, 25000)),
    ("吉林省", random.randint(2000, 15000)),
    ("黑龙江省", random.randint(2000, 18000)),
    ("上海市", random.randint(8000, 45000)),
    ("江苏省", random.randint(10000, 50000)),
    ("浙江省", random.randint(8000, 45000)),
    ("安徽省", random.randint(4000, 25000)),
    ("福建省", random.randint(5000, 30000)),
    ("江西省", random.randint(3000, 20000)),
    ("山东省", random.randint(8000, 40000)),
    ("河南省", random.randint(6000, 35000)),
    ("湖北省", random.randint(5000, 30000)),
    ("湖南省", random.randint(4000, 28000)),
    ("广东省", random.randint(12000, 55000)),
    ("广西壮族自治区", random.randint(3000, 18000)),
    ("海南省", random.randint(1500, 8000)),
    ("重庆市", random.randint(3000, 20000)),
    ("四川省", random.randint(6000, 32000)),
    ("贵州省", random.randint(2000, 12000)),
    ("云南省", random.randint(2500, 15000)),
    ("西藏自治区", random.randint(500, 3000)),
    ("陕西省", random.randint(3500, 22000)),
    ("甘肃省", random.randint(1500, 10000)),
    ("青海省", random.randint(800, 5000)),
    ("宁夏回族自治区", random.randint(600, 4000)),
    ("新疆维吾尔自治区", random.randint(1200, 8000)),
    ("台湾省", random.randint(3000, 18000)),
    ("香港特别行政区", random.randint(5000, 25000)),
    ("澳门特别行政区", random.randint(1000, 5000)),
]

china_map = (
    Map(init_opts=opts.InitOpts(
        width="1000px",
        height="700px",
        theme=ThemeType.LIGHT,
    ))
    .add(
        "GDP",
        province_data,
        "china",
        is_map_symbol_show=False,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="中国各省GDP分布 (模拟数据)", subtitle="单位: 亿元"),
        visualmap_opts=opts.VisualMapOpts(
            min_=500,
            max_=55000,
            is_piecewise=True,
            pieces=[
                {"min": 40000, "label": ">40000", "color": "#800026"},
                {"min": 25000, "max": 40000, "label": "25000-40000", "color": "#BD0026"},
                {"min": 15000, "max": 25000, "label": "15000-25000", "color": "#E31A1C"},
                {"min": 8000, "max": 15000, "label": "8000-15000", "color": "#FC4E2A"},
                {"min": 3000, "max": 8000, "label": "3000-8000", "color": "#FD8D3C"},
                {"min": 0, "max": 3000, "label": "<3000", "color": "#FED976"},
            ],
        ),
        tooltip_opts=opts.TooltipOpts(
            formatter="{b}<br/>GDP: {c} 亿元",
        ),
    )
)

china_map.render("output_pyecharts_中国地图.html")
print("中国省级地图已保存为 output_pyecharts_中国地图.html")

# ============================================================
# 2. 省内城市地图 - 以广东省为例
# ============================================================
print("\n--- 2. 省内城市地图 ---")

gd_city_data = [
    ("广州市", random.randint(5000, 25000)),
    ("深圳市", random.randint(6000, 30000)),
    ("珠海市", random.randint(1000, 5000)),
    ("汕头市", random.randint(800, 3000)),
    ("佛山市", random.randint(2000, 10000)),
    ("韶关市", random.randint(500, 2000)),
    ("湛江市", random.randint(800, 3500)),
    ("肇庆市", random.randint(600, 2500)),
    ("江门市", random.randint(700, 3000)),
    ("茂名市", random.randint(600, 2800)),
    ("惠州市", random.randint(1000, 4500)),
    ("梅州市", random.randint(400, 1500)),
    ("汕尾市", random.randint(300, 1200)),
    ("河源市", random.randint(300, 1100)),
    ("阳江市", random.randint(350, 1300)),
    ("清远市", random.randint(400, 1600)),
    ("东莞市", random.randint(2000, 9000)),
    ("中山市", random.randint(1000, 5000)),
    ("潮州市", random.randint(300, 1200)),
    ("揭阳市", random.randint(500, 2000)),
    ("云浮市", random.randint(250, 900)),
]

gd_map = (
    Map(init_opts=opts.InitOpts(
        width="900px",
        height="700px",
        theme=ThemeType.MACARONS,
    ))
    .add(
        "GDP",
        gd_city_data,
        "广东",
        is_map_symbol_show=False,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="广东省各市GDP分布 (模拟数据)"),
        visualmap_opts=opts.VisualMapOpts(
            min_=0,
            max_=30000,
            is_piecewise=False,
        ),
    )
)

gd_map.render("output_pyecharts_广东地图.html")
print("广东省城市地图已保存为 output_pyecharts_广东地图.html")

# ============================================================
# 3. 地理坐标图 - Geo()
# ============================================================
print("\n--- 3. 地理坐标图 Geo() ---")

geo_data = [
    ("广州", random.randint(100, 500)),
    ("北京", random.randint(100, 500)),
    ("上海", random.randint(100, 500)),
    ("成都", random.randint(100, 500)),
    ("武汉", random.randint(100, 500)),
    ("杭州", random.randint(100, 500)),
    ("重庆", random.randint(100, 500)),
    ("南京", random.randint(100, 500)),
    ("西安", random.randint(100, 500)),
    ("长沙", random.randint(100, 500)),
    ("深圳", random.randint(100, 500)),
    ("郑州", random.randint(100, 500)),
    ("天津", random.randint(100, 500)),
    ("沈阳", random.randint(100, 500)),
    ("哈尔滨", random.randint(100, 500)),
]

geo = (
    Geo(init_opts=opts.InitOpts(
        width="900px",
        height="700px",
        theme=ThemeType.DARK,
    ))
    .add_schema(
        maptype="china",
        itemstyle_opts=opts.ItemStyleOpts(color="#323c48", border_color="#111"),
    )
    .add(
        "热度",
        geo_data,
        type_=ChartType.EFFECT_SCATTER,
        symbol_size=8,
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="主要城市热度分布 (模拟数据)"),
        visualmap_opts=opts.VisualMapOpts(
            min_=100,
            max_=500,
            is_piecewise=True,
        ),
    )
)

geo.render("output_pyecharts_地理坐标图.html")
print("地理坐标图已保存为 output_pyecharts_地理坐标图.html")

# ============================================================
# 4. 热力图 - HeatMap()
# ============================================================
print("\n--- 4. 热力图 HeatMap() ---")

hours = [f"{i}:00" for i in range(24)]
days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

heat_data = []
for i, day in enumerate(days):
    for j, hour in enumerate(hours):
        if 9 <= j <= 17 and i < 5:
            value = random.randint(50, 100)
        elif (j >= 19 or j <= 2) and i >= 5:
            value = random.randint(40, 90)
        else:
            value = random.randint(5, 40)
        heat_data.append([j, i, value])

heatmap = (
    HeatMap(init_opts=opts.InitOpts(
        width="1000px",
        height="500px",
        theme=ThemeType.LIGHT,
    ))
    .add_xaxis(hours)
    .add_yaxis(
        "访问量",
        days,
        heat_data,
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="网站访问量热力图 (模拟数据)", subtitle="按小时和星期统计"),
        xaxis_opts=opts.AxisOpts(name="时间", type_="category"),
        yaxis_opts=opts.AxisOpts(name="星期", type_="category"),
        visualmap_opts=opts.VisualMapOpts(
            min_=0,
            max_=100,
            is_calculable=True,
            orient="horizontal",
            pos_top="5%",
            pos_left="center",
        ),
    )
)

heatmap.render("output_pyecharts_热力图.html")
print("热力图已保存为 output_pyecharts_热力图.html")

# ============================================================
# 5. render_embed() 概念说明
# ============================================================
print("\n--- 5. render_embed() 说明 ---")
print("""
render_embed() 的用途:
- 将图表渲染为HTML字符串，可嵌入到Flask/Django等Web框架中
- 与 render() 的区别:
    render("file.html")  -> 保存为独立HTML文件
    render_embed()       -> 返回HTML字符串，用于嵌入页面

示例 (Flask集成):
    from flask import Flask, render_template_string
    from pyecharts.charts import Bar

    app = Flask(__name__)

    @app.route("/")
    def index():
        bar = Bar()
        bar.add_xaxis(["A", "B", "C"])
        bar.add_yaxis("系列", [1, 2, 3])
        return render_template_string(
            '<html><body>{{ chart|safe }}</body></html>',
            chart=bar.render_embed()
        )

    app.run()
""")

print("\n" + "=" * 60)
print("Pyecharts 地图与热力图演示完成！")
print("所有HTML文件已保存到当前目录，可在浏览器中打开查看交互效果。")
print("=" * 60)
