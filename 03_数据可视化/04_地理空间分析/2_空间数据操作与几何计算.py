# 数据来源: geopandas内置naturalearth数据集、自建模拟数据

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# === 1. 缓冲区分析 ===
print("=== 缓冲区分析 ===")
cities_data = {
    "城市": ["北京", "上海", "广州"],
    "人口(万)": [2189, 2487, 1868],
    "geometry": [Point(116.4, 39.9), Point(121.5, 31.2), Point(113.3, 23.1)],
}
gdf = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")

# 转投影坐标系后计算缓冲区（单位：米）
gdf_proj = gdf.to_crs("EPSG:3857")
buffer_50km = gdf_proj.buffer(50000)
buffer_gdf = gpd.GeoDataFrame(gdf[["城市"]], geometry=buffer_50km, crs="EPSG:3857")
buffer_gdf = buffer_gdf.to_crs("EPSG:4326")
print("50km缓冲区面积（平方公里）:")
for _, row in buffer_gdf.to_crs("EPSG:3857").iterrows():
    area_km2 = row.geometry.area / 1e6
    print(f"  {row['城市']}: {area_km2:.0f} km²")

# === 2. 空间关系判断 ===
print("\n=== 空间关系判断 ===")
polygon_beijing = Polygon([(115.5, 39.5), (117.5, 39.5), (117.5, 41.0), (115.5, 41.0)])
gdf_region = gpd.GeoDataFrame({"区域": ["北京周边"]}, geometry=[polygon_beijing], crs="EPSG:4326")

point_in = Point(116.8, 40.2)
point_out = Point(121.5, 31.2)

print(f"点{point_in} 在区域内: {point_in.within(polygon_beijing)}")
print(f"点{point_out} 在区域内: {point_out.within(polygon_beijing)}")
print(f"区域 包含 点: {polygon_beijing.contains(point_in)}")
print(f"区域 相交 点: {polygon_beijing.intersects(Point(117.5, 40.0))}")

# GeoDataFrame批量空间查询
gdf["在北京周边"] = gdf.geometry.within(polygon_beijing)
print("\n各城市是否在北京周边:")
print(gdf[["城市", "在北京周边"]])

# === 3. 面积与距离计算 ===
print("\n=== 面积与距离计算 ===")
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
china = world[world["name"] == "China"].to_crs("EPSG:3857")
area_km2 = china.geometry.iloc[0].area / 1e6
print(f"中国面积（近似）: {area_km2:.0f} km²")

# 城市间距离
gdf_proj = gdf.to_crs("EPSG:3857")
dist_beijing_shanghai = gdf_proj.geometry.iloc[0].distance(gdf_proj.geometry.iloc[1]) / 1000
dist_beijing_guangzhou = gdf_proj.geometry.iloc[0].distance(gdf_proj.geometry.iloc[2]) / 1000
print(f"北京-上海距离: {dist_beijing_shanghai:.0f} km")
print(f"北京-广州距离: {dist_beijing_guangzhou:.0f} km")

# 距离矩阵
from itertools import combinations
print("\n城市间距离矩阵（km）:")
cities = gdf["城市"].tolist()
print(f"{'':>6}", end="")
for c in cities:
    print(f"{c:>10}", end="")
print()
for i, c1 in enumerate(cities):
    print(f"{c1:>6}", end="")
    for j, c2 in enumerate(cities):
        if i == j:
            print(f"{'0':>10}", end="")
        else:
            d = gdf_proj.geometry.iloc[i].distance(gdf_proj.geometry.iloc[j]) / 1000
            print(f"{d:>10.0f}", end="")
    print()

# === 4. 空间连接 ===
print("\n=== 空间连接 ===")
regions_data = {
    "区域": ["华北", "华东", "华南"],
    "geometry": [
        Polygon([(110, 35), (120, 35), (120, 42), (110, 42)]),
        Polygon([(118, 28), (123, 28), (123, 35), (118, 35)]),
        Polygon([(109, 20), (117, 20), (117, 27), (109, 27)]),
    ],
}
gdf_regions = gpd.GeoDataFrame(regions_data, crs="EPSG:4326")

more_cities = {
    "城市": ["北京", "上海", "广州", "深圳", "杭州", "天津", "武汉"],
    "geometry": [
        Point(116.4, 39.9), Point(121.5, 31.2), Point(113.3, 23.1),
        Point(114.1, 22.5), Point(120.2, 30.3), Point(117.2, 39.1), Point(114.3, 30.6),
    ],
}
gdf_more = gpd.GeoDataFrame(more_cities, crs="EPSG:4326")

joined = gpd.sjoin(gdf_more, gdf_regions, how="left", predicate="within")
print("城市-区域空间连接结果:")
print(joined[["城市", "区域"]])
