# 数据来源: 自建模拟数据

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# === 模拟城市POI点数据 ===
center_lat, center_lon = 31.23, 121.47
n_points = 500

lat = np.random.normal(center_lat, 0.05, n_points)
lon = np.random.normal(center_lon, 0.06, n_points)
weights = np.random.uniform(1, 100, n_points)

poi_data = pd.DataFrame({"纬度": lat, "经度": lon, "权重": weights})
print(f"生成POI点数: {len(poi_data)}")
print(poi_data.describe())

# === 1. Folium热力图 ===
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
heat_data = [[row["纬度"], row["经度"], row["权重"]] for _, row in poi_data.iterrows()]

HeatMap(
    heat_data,
    min_opacity=0.2,
    max_zoom=15,
    radius=15,
    blur=20,
    gradient={0.0: "blue", 0.3: "cyan", 0.5: "lime", 0.7: "yellow", 1.0: "red"},
).add_to(m)
m.save(output_dir / "density_heatmap.html")
print("已保存: density_heatmap.html")

# === 2. 点聚合（MarkerCluster） ===
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
marker_cluster = MarkerCluster(name="POI聚合").add_to(m)

for _, row in poi_data.iterrows():
    folium.CircleMarker(
        location=[row["纬度"], row["经度"]],
        radius=3,
        color="blue",
        fill=True,
        fill_opacity=0.5,
    ).add_to(marker_cluster)

folium.LayerControl().add_to(m)
m.save(output_dir / "density_marker_cluster.html")
print("已保存: density_marker_cluster.html")

# === 3. 六边形密度图（静态） ===
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

hb = ax.hexbin(
    poi_data["经度"], poi_data["纬度"],
    C=poi_data["权重"],
    gridsize=25,
    cmap="YlOrRd",
    reduce_C_function=np.mean,
    mincnt=1,
)
cb = fig.colorbar(hb, ax=ax, label="平均权重")
ax.set_xlabel("经度")
ax.set_ylabel("纬度")
ax.set_title("POI六边形密度图", fontsize=14)
fig.savefig(output_dir / "hexbin_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: hexbin_density.png")

# === 4. 核密度估计图 ===
from scipy.stats import gaussian_kde

xy = np.vstack([poi_data["纬度"], poi_data["经度"]])
kde = gaussian_kde(xy, weights=poi_data["权重"])

grid_res = 100
lat_grid = np.linspace(lat.min() - 0.02, lat.max() + 0.02, grid_res)
lon_grid = np.linspace(lon.min() - 0.02, lon.max() + 0.02, grid_res)
lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
positions = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])
density = kde(positions).reshape(grid_res, grid_res)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
contour = ax.contourf(lon_mesh, lat_mesh, density, levels=20, cmap="hot_r")
ax.scatter(poi_data["经度"], poi_data["纬度"], s=2, c="blue", alpha=0.3, label="POI点")
fig.colorbar(contour, ax=ax, label="密度")
ax.set_xlabel("经度")
ax.set_ylabel("纬度")
ax.set_title("POI核密度估计", fontsize=14)
ax.legend()
fig.savefig(output_dir / "kde_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: kde_density.png")

# === 5. 网格密度统计 ===
lat_bins = pd.cut(poi_data["纬度"], bins=10)
lon_bins = pd.cut(poi_data["经度"], bins=10)
grid_count = poi_data.groupby([lat_bins, lon_bins]).size().unstack(fill_value=0)

print("\n网格密度统计（每格POI数量）:")
print(grid_count)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
im = ax.imshow(grid_count.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(grid_count.columns)))
ax.set_xticklabels([f"{x.mid:.3f}" for x in grid_count.columns], rotation=45, fontsize=8)
ax.set_yticks(range(len(grid_count.index)))
ax.set_yticklabels([f"{x.mid:.3f}" for x in grid_count.index], fontsize=8)
fig.colorbar(im, ax=ax, label="POI数量")
ax.set_title("网格密度统计", fontsize=14)
ax.set_xlabel("经度区间")
ax.set_ylabel("纬度区间")
fig.savefig(output_dir / "grid_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: grid_density.png")

print("\n所有密度可视化文件已保存至 output 目录")
