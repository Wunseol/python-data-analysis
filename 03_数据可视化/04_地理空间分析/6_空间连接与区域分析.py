# 数据来源: geopandas内置naturalearth数据集、自建模拟数据

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

np.random.seed(42)

# === 1. 构建模拟区域与点数据 ===
regions_data = {
    "区域": ["朝阳区", "海淀区", "西城区", "东城区", "丰台区"],
    "geometry": [
        Polygon([(116.4, 39.8), (116.7, 39.8), (116.7, 40.0), (116.4, 40.0)]),
        Polygon([(116.1, 39.9), (116.4, 39.9), (116.4, 40.1), (116.1, 40.1)]),
        Polygon([(116.3, 39.88), (116.4, 39.88), (116.4, 39.95), (116.3, 39.95)]),
        Polygon([(116.4, 39.88), (116.5, 39.88), (116.5, 39.95), (116.4, 39.95)]),
        Polygon([(116.1, 39.78), (116.4, 39.78), (116.4, 39.9), (116.1, 39.9)]),
    ],
}
gdf_regions = gpd.GeoDataFrame(regions_data, crs="EPSG:4326")

n_poi = 200
poi_lats = np.random.uniform(39.78, 40.1, n_poi)
poi_lons = np.random.uniform(116.1, 116.7, n_poi)
poi_types = np.random.choice(["餐饮", "购物", "教育", "医疗", "交通"], n_poi)

pois = gpd.GeoDataFrame(
    {"类型": poi_types},
    geometry=[Point(lon, lat) for lon, lat in zip(poi_lons, poi_lats)],
    crs="EPSG:4326",
)
print(f"区域数: {len(gdf_regions)}, POI数: {len(pois)}")

# === 2. sjoin空间连接 ===
print("\n=== sjoin空间连接 ===")
joined = gpd.sjoin(pois, gdf_regions, how="inner", predicate="within")
print(f"匹配到区域的POI数: {len(joined)}")
print("\n各区域POI数量:")
print(joined.groupby("区域").size())

# === 3. 点在多边形内统计 ===
print("\n=== 点在多边形内统计 ===")
stats = joined.groupby("区域")["类型"].value_counts().unstack(fill_value=0)
print("各区域各类型POI统计:")
print(stats)

total_by_region = joined.groupby("区域").size().reset_index(name="POI总数")
gdf_regions = gdf_regions.merge(total_by_region, on="区域", how="left").fillna(0)
print("\n区域POI密度:")
gdf_regions_proj = gdf_regions.to_crs("EPSG:3857")
gdf_regions["面积(km²)"] = gdf_regions_proj.geometry.area / 1e6
gdf_regions["POI密度"] = gdf_regions["POI总数"] / gdf_regions["面积(km²)"]
print(gdf_regions[["区域", "POI总数", "面积(km²)", "POI密度"]])

# === 4. 区域聚合分析 ===
print("\n=== 区域聚合分析 ===")
region_agg = joined.dissolve(by="区域", aggfunc={"类型": "count"})
region_agg = region_agg.rename(columns={"类型": "POI数量"})
print(region_agg[["POI数量"]])

# 按类型聚合
type_agg = joined.groupby("类型").size().reset_index(name="数量")
type_agg["占比"] = type_agg["数量"] / type_agg["数量"].sum() * 100
print("\nPOI类型分布:")
print(type_agg.to_string(index=False))

# === 5. 最近邻分析 ===
print("\n=== 最近邻分析 ===")
from scipy.spatial import cKDTree

coords = np.array(list(zip(poi_lons, poi_lats)))
tree = cKDTree(coords)

distances, indices = tree.query(coords, k=2)
nn_distances = distances[:, 1]

print(f"最近邻距离统计（度）:")
print(f"  均值: {nn_distances.mean():.6f}")
print(f"  中位数: {np.median(nn_distances):.6f}")
print(f"  最小值: {nn_distances.min():.6f}")
print(f"  最大值: {nn_distances.max():.6f}")

# 计算最近邻指数
area = gdf_regions_proj.unary_union.area / 1e6
expected_nn = 0.5 * np.sqrt(area / len(pois))
observed_nn = np.mean(nn_distances) * 111  # 粗略转换为km
nn_index = observed_nn / (expected_nn * 111)
print(f"\n最近邻指数(R): {nn_index:.4f}")
print("  R<1: 聚集分布 | R≈1: 随机分布 | R>1: 均匀分布")

# === 6. 保存分析结果 ===
result_path = output_dir / "spatial_analysis_result.geojson"
gdf_regions.to_file(result_path, driver="GeoJSON")
print(f"\n分析结果已保存至: {result_path}")
