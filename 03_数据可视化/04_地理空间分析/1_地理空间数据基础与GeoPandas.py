# 数据来源: geopandas内置naturalearth数据集、自建模拟数据
# 依赖库最低版本要求: geopandas>=0.14, folium>=0.15

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# === 1. 从内置数据集创建GeoDataFrame ===
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
print("世界地图数据集:")
print(world.head())
print(f"\n几何类型: {world.geom_type.unique()}")
print(f"CRS: {world.crs}")

# === 2. 从字典创建GeoDataFrame ===
cities_data = {
    "城市": ["北京", "上海", "广州", "深圳", "成都"],
    "人口(万)": [2189, 2487, 1868, 1756, 2094],
    "经度": [116.4, 121.5, 113.3, 114.1, 104.1],
    "纬度": [39.9, 31.2, 23.1, 22.5, 30.6],
}
df = pd.DataFrame(cities_data)
geometry = [Point(lon, lat) for lon, lat in zip(df["经度"], df["纬度"])]
gdf_cities = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
print("\n城市GeoDataFrame:")
print(gdf_cities)

# === 3. 几何类型演示 ===
# Point
point = Point(116.4, 39.9)
print(f"\nPoint: {point}, 类型: {point.geom_type}")

# LineString
line = LineString([(116.4, 39.9), (121.5, 31.2), (113.3, 23.1)])
print(f"LineString: {line}, 长度(度): {line.length:.2f}")

# Polygon
polygon = Polygon([(116, 39), (117, 39), (117, 40), (116, 40)])
print(f"Polygon: {polygon}, 面积(平方度): {polygon.area:.4f}")

# === 4. GeoSeries操作 ===
print("\n--- GeoSeries操作 ---")
gs = gdf_cities.geometry
print(f"质心:\n{gs.centroid}")
print(f"\n边界:\n{gs.boundary}")
print(f"\n包络矩形:\n{gs.envelope}")

# === 5. CRS坐标参考系 ===
print("\n--- CRS坐标参考系 ---")
print(f"原始CRS: {gdf_cities.crs}")

# 转换为投影坐标系（适合距离/面积计算）
gdf_projected = gdf_cities.to_crs("EPSG:3857")
print(f"投影后CRS: {gdf_projected.crs}")
print(f"投影后坐标示例（北京）:\n{gdf_projected.geometry.iloc[0]}")

# === 6. 保存与读取 ===
save_path = output_dir / "cities.geojson"
gdf_cities.to_file(save_path, driver="GeoJSON")
print(f"\n已保存至: {save_path}")

gdf_loaded = gpd.read_file(save_path)
print(f"重新加载后记录数: {len(gdf_loaded)}")
