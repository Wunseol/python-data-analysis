# 数据来源: 自建模拟数据

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ============================================================
# 第一步：模拟城市POI数据
# ============================================================
print("=" * 50)
print("第一步：模拟城市POI数据")
print("=" * 50)

city_center = {"lat": 30.57, "lon": 104.07, "name": "成都"}

districts = {
    "锦江区": {"center": (30.62, 104.08), "spread": 0.02},
    "青羊区": {"center": (30.67, 104.03), "spread": 0.025},
    "金牛区": {"center": (30.70, 104.05), "spread": 0.02},
    "武侯区": {"center": (30.63, 104.04), "spread": 0.025},
    "成华区": {"center": (30.68, 104.10), "spread": 0.02},
    "高新区": {"center": (30.58, 104.07), "spread": 0.03},
}

poi_types = {
    "餐饮": {"weight": 0.35, "score_range": (3.0, 5.0)},
    "购物": {"weight": 0.20, "score_range": (3.2, 4.8)},
    "教育": {"weight": 0.15, "score_range": (3.5, 4.9)},
    "医疗": {"weight": 0.10, "score_range": (3.0, 4.7)},
    "交通": {"weight": 0.20, "score_range": (3.3, 4.6)},
}

records = []
for district, info in districts.items():
    n = np.random.randint(40, 80)
    for _ in range(n):
        ptype = np.random.choice(
            list(poi_types.keys()),
            p=[poi_types[t]["weight"] for t in poi_types],
        )
        lat = np.random.normal(info["center"][0], info["spread"])
        lon = np.random.normal(info["center"][1], info["spread"])
        score = np.random.uniform(*poi_types[ptype]["score_range"])
        records.append({
            "区域": district,
            "类型": ptype,
            "纬度": lat,
            "经度": lon,
            "评分": round(score, 1),
        })

df_poi = pd.DataFrame(records)
print(f"生成POI总数: {len(df_poi)}")
print(df_poi.head())

gdf_poi = gpd.GeoDataFrame(
    df_poi,
    geometry=[Point(r["经度"], r["纬度"]) for _, r in df_poi.iterrows()],
    crs="EPSG:4326",
)

# ============================================================
# 第二步：空间分布分析
# ============================================================
print("\n" + "=" * 50)
print("第二步：空间分布分析")
print("=" * 50)

dist_stats = df_poi.groupby("区域").agg(
    POI数量=("类型", "count"),
    平均评分=("评分", "mean"),
).round(2)
print("\n各区域POI统计:")
print(dist_stats)

type_stats = df_poi.groupby("类型").agg(
    数量=("区域", "count"),
    平均评分=("评分", "mean"),
).round(2)
type_stats["占比(%)"] = (type_stats["数量"] / type_stats["数量"].sum() * 100).round(1)
print("\n各类型POI统计:")
print(type_stats)

# ============================================================
# 第三步：热力图
# ============================================================
print("\n" + "=" * 50)
print("第三步：热力图")
print("=" * 50)

m = folium.Map(
    location=[city_center["lat"], city_center["lon"]],
    zoom_start=12,
    tiles="CartoDB positron",
)

heat_data = [[r["纬度"], r["经度"], r["评分"]] for _, r in df_poi.iterrows()]
HeatMap(
    heat_data,
    min_opacity=0.3,
    radius=12,
    blur=15,
    gradient={0.0: "blue", 0.3: "cyan", 0.5: "lime", 0.7: "yellow", 1.0: "red"},
).add_to(m)
m.save(output_dir / "city_heatmap.html")
print("已保存: city_heatmap.html")

# ============================================================
# 第四步：区域统计
# ============================================================
print("\n" + "=" * 50)
print("第四步：区域统计")
print("=" * 50)

district_polys = {}
for name, info in districts.items():
    cx, cy = info["center"]
    s = info["spread"] * 1.5
    district_polys[name] = Polygon([
        (cy - s, cx - s), (cy + s, cx - s),
        (cy + s, cx + s), (cy - s, cx + s),
    ])

gdf_districts = gpd.GeoDataFrame(
    {"区域": list(district_polys.keys())},
    geometry=list(district_polys.values()),
    crs="EPSG:4326",
)

joined = gpd.sjoin(gdf_poi, gdf_districts, how="inner", predicate="within")
region_poi_count = joined.groupby("区域_left").size().reset_index(name="POI数量")
region_poi_count.columns = ["区域", "POI数量"]

gdf_districts = gdf_districts.merge(region_poi_count, on="区域", how="left").fillna(0)
gdf_districts_proj = gdf_districts.to_crs("EPSG:3857")
gdf_districts["面积(km²)"] = (gdf_districts_proj.geometry.area / 1e6).round(2)
gdf_districts["POI密度"] = (gdf_districts["POI数量"] / gdf_districts["面积(km²)"]).round(1)

print("\n区域统计结果:")
print(gdf_districts[["区域", "POI数量", "面积(km²)", "POI密度"]])

# 各区域各类型POI交叉统计
cross_stats = joined.groupby(["区域_left", "类型"]).size().unstack(fill_value=0)
cross_stats.index.name = "区域"
print("\n区域×类型交叉统计:")
print(cross_stats)

# ============================================================
# 第五步：交互地图输出
# ============================================================
print("\n" + "=" * 50)
print("第五步：交互地图输出")
print("=" * 50)

m = folium.Map(
    location=[city_center["lat"], city_center["lon"]],
    zoom_start=12,
)

# 区域着色图层
choropleth = folium.Choropleth(
    geo_data=gdf_districts.to_json(),
    data=gdf_districts,
    columns=["区域", "POI密度"],
    key_on="feature.properties.区域",
    fill_color="YlOrRd",
    fill_opacity=0.5,
    line_opacity=0.8,
    legend_name="POI密度（个/km²）",
    name="区域POI密度",
).add_to(m)

for _, row in gdf_districts.iterrows():
    folium.GeoJson(
        row["geometry"],
        style_function=lambda x: {"fillOpacity": 0, "color": "transparent"},
        tooltip=f"{row['区域']}: POI密度={row['POI密度']}个/km²",
    ).add_to(m)

# POI点聚合图层
poi_group = folium.FeatureGroup(name="POI分布")
marker_cluster = MarkerCluster().add_to(poi_group)

type_colors = {"餐饮": "red", "购物": "blue", "教育": "green", "医疗": "purple", "交通": "orange"}
for _, row in df_poi.iterrows():
    folium.CircleMarker(
        location=[row["纬度"], row["经度"]],
        radius=3,
        color=type_colors.get(row["类型"], "gray"),
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['类型']} | 评分: {row['评分']} | {row['区域']}",
    ).add_to(marker_cluster)

poi_group.add_to(m)

# 热力图图层
heat_group = folium.FeatureGroup(name="评分热力图")
HeatMap(heat_data, radius=15, blur=20, min_opacity=0.2).add_to(heat_group)
heat_group.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

output_path = output_dir / "city_comprehensive_map.html"
m.save(output_path)
print(f"综合交互地图已保存至: {output_path}")

# 静态统计图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

dist_stats["POI数量"].plot.bar(ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("各区域POI数量", fontsize=13)
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=0)

type_stats["数量"].plot.pie(
    ax=axes[1], autopct="%1.1f%%", colors=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"],
    startangle=90,
)
axes[1].set_title("POI类型分布", fontsize=13)
axes[1].set_ylabel("")

fig.tight_layout()
fig.savefig(output_dir / "city_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: city_stats.png")

print("\n综合案例分析完成！所有文件已保存至 output 目录")
