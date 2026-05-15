# 数据来源: Plotly内置 gapminder 数据集 + 自造模拟数据
# 地理地图可视化是数据可视化中非常重要的一类图表
# Plotly 支持: 等值区域图(choropleth)、地理散点图(scatter_geo)、Mapbox地图

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

print("=" * 60)
print("Plotly 地理地图可视化")
print("=" * 60)

# ============================================================
# 1. 等值区域图 - px.choropleth() 基础
# ============================================================
print("\n--- 1. 等值区域图 px.choropleth() ---")

df_gap = px.data.gapminder()
df_2007 = df_gap[df_gap["year"] == 2007]

fig_choro = px.choropleth(
    df_2007,
    locations="iso_alpha",
    color="lifeExp",
    hover_name="country",
    color_continuous_scale=px.colors.sequential.Plasma,
    title="2007年全球各国预期寿命",
    labels={"lifeExp": "预期寿命 (岁)"},
)

fig_choro.update_layout(margin=dict(l=0, r=0, t=40, b=0))

fig_choro.show()
fig_choro.write_html("output_等值区域图_全球.html")
print("全球等值区域图已保存")

# ============================================================
# 2. 限定范围的等值区域图 - scope 参数
# ============================================================
print("\n--- 2. 限定范围: scope 参数 ---")

fig_asia = px.choropleth(
    df_2007,
    locations="iso_alpha",
    color="pop",
    hover_name="country",
    scope="asia",
    color_continuous_scale=px.colors.sequential.YlOrRd,
    title="2007年亚洲各国人口",
    labels={"pop": "人口"},
)

fig_asia.update_layout(margin=dict(l=0, r=0, t=40, b=0))

fig_asia.show()
fig_asia.write_html("output_等值区域图_亚洲.html")
print("亚洲等值区域图已保存")

# ============================================================
# 3. 投影方式 - projection 参数
# ============================================================
print("\n--- 3. 投影方式 projection ---")

projections = ["natural earth", "mercator", "orthographic", "robinson"]

for proj in projections:
    fig_proj = px.choropleth(
        df_2007,
        locations="iso_alpha",
        color="gdpPercap",
        hover_name="country",
        projection=proj,
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"投影方式: {proj}",
    )

    fig_proj.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    fig_proj.show()
    fig_proj.write_html(f"output_投影_{proj.replace(' ', '_')}.html")
    print(f"  投影 '{proj}' 已保存")

# ============================================================
# 4. 地理散点图 - px.scatter_geo()
# ============================================================
print("\n--- 4. 地理散点图 px.scatter_geo() ---")

fig_scatter_geo = px.scatter_geo(
    df_2007,
    locations="iso_alpha",
    size="pop",
    color="continent",
    hover_name="country",
    size_max=50,
    projection="natural earth",
    title="2007年全球人口分布 (地理散点图)",
)

fig_scatter_geo.update_layout(margin=dict(l=0, r=0, t=40, b=0))

fig_scatter_geo.show()
fig_scatter_geo.write_html("output_地理散点图.html")
print("地理散点图已保存")

# ============================================================
# 5. 中国省级地图 - 使用自造模拟数据
# ============================================================
print("\n--- 5. 中国省级地图 ---")

china_provinces = [
    "北京市", "天津市", "河北省", "山西省", "内蒙古自治区",
    "辽宁省", "吉林省", "黑龙江省", "上海市", "江苏省",
    "浙江省", "安徽省", "福建省", "江西省", "山东省",
    "河南省", "湖北省", "湖南省", "广东省", "广西壮族自治区",
    "海南省", "重庆市", "四川省", "贵州省", "云南省",
    "西藏自治区", "陕西省", "甘肃省", "青海省", "宁夏回族自治区",
    "新疆维吾尔自治区", "台湾省", "香港特别行政区", "澳门特别行政区",
]

np.random.seed(42)
df_china = pd.DataFrame({
    "省份": china_provinces,
    "GDP (亿元)": np.random.randint(1000, 120000, len(china_provinces)),
    "人口 (万人)": np.random.randint(200, 12000, len(china_provinces)),
})

fig_china = px.choropleth(
    df_china,
    locations="省份",
    color="GDP (亿元)",
    scope="asia",
    title="中国各省GDP模拟数据",
    color_continuous_scale=px.colors.sequential.Reds,
)

fig_china.update_geos(
    center=dict(lat=35, lon=105),
    projection_scale=3,
)

fig_china.update_layout(margin=dict(l=0, r=0, t=40, b=0))

fig_china.show()
fig_china.write_html("output_中国地图.html")
print("中国省级地图已保存 (注意: 中文省份名需要Plotly支持中文地理编码)")

# ============================================================
# 6. 使用 graph_objects 创建精细地理图
# ============================================================
print("\n--- 6. graph_objects 精细地理图 ---")

fig_go = go.Figure(go.Choropleth(
    locations=df_2007["iso_alpha"],
    z=df_2007["gdpPercap"],
    text=df_2007["country"],
    colorscale="Blues",
    autocolorscale=False,
    reversescale=True,
    marker_line_color="darkgray",
    marker_line_width=0.5,
    colorbar_title="人均GDP<br>(美元)",
))

fig_go.update_layout(
    title_text="2007年全球人均GDP (graph_objects)",
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth",
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig_go.show()
fig_go.write_html("output_地理图_go.html")
print("graph_objects地理图已保存")

# ============================================================
# 7. Mapbox 地图概念说明 (需要 Mapbox token)
# ============================================================
print("\n--- 7. Mapbox 地图 (需要Token) ---")
print("""
Mapbox 地图是 Plotly 中的另一种地理可视化方式，特点:
1. 需要注册 Mapbox 账号获取 access_token
2. 支持真实的街道级地图底图
3. 使用 px.scatter_mapbox() 或 px.choropleth_mapbox()
4. 注册地址: https://account.mapbox.com/auth/signup/

示例代码 (需要有效token):
    px.set_mapbox_access_token("your_token_here")
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon",
        size="value", color="category",
        mapbox_style="open-street-map",
    )
    fig.show()
""")

print("\n" + "=" * 60)
print("Plotly 地理地图可视化演示完成！")
print("=" * 60)
