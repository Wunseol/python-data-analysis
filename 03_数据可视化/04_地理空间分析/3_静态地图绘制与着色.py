# 数据来源: geopandas内置naturalearth数据集

import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# === 1. 基础地图绘制 ===
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.5)
ax.set_title("世界地图", fontsize=16)
ax.set_axis_off()
fig.savefig(output_dir / "basic_world_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: basic_world_map.png")

# === 2. Choropleth着色地图 ===
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
world.plot(
    column="pop_est",
    ax=ax,
    legend=True,
    legend_kwds={
        "label": "人口",
        "orientation": "horizontal",
        "shrink": 0.6,
        "fmt": "%.0f",
    },
    cmap="YlOrRd",
    missing_kwds={"color": "lightgrey"},
)
ax.set_title("世界各国人口分布", fontsize=16)
ax.set_axis_off()
fig.savefig(output_dir / "choropleth_population.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: choropleth_population.png")

# === 3. 分类着色 ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
schemes = ["equal_interval", "quantiles", "natural_breaks", "fisher_jenks"]
titles = ["等间距分类", "分位数分类", "自然断点分类", "Fisher-Jenks分类"]

for ax, scheme, title in zip(axes.flatten(), schemes, titles):
    world.plot(
        column="gdp_md_est",
        ax=ax,
        legend=True,
        scheme=scheme,
        k=5,
        cmap="Blues",
        legend_kwds={"fmt": "%.0f", "loc": "lower left"},
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()

fig.suptitle("GDP分类着色方案对比", fontsize=16, y=1.0)
fig.tight_layout()
fig.savefig(output_dir / "classification_schemes.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: classification_schemes.png")

# === 4. 配色方案对比 ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cmaps = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"]

for ax, cmap in zip(axes.flatten(), cmaps):
    world.plot(column="pop_est", ax=ax, legend=True, cmap=cmap,
               legend_kwds={"shrink": 0.5, "fmt": "%.0f"})
    ax.set_title(f"cmap={cmap}", fontsize=12)
    ax.set_axis_off()

fig.suptitle("配色方案对比", fontsize=16)
fig.tight_layout()
fig.savefig(output_dir / "colormap_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: colormap_comparison.png")

# === 5. 多图层叠加 ===
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
asia = world[world["continent"] == "Asia"]
asia.plot(ax=ax, column="pop_est", cmap="Oranges", legend=True,
          legend_kwds={"label": "人口", "shrink": 0.6},
          edgecolor="grey", linewidth=0.5)

cities = gpd.read_file(gpd.datasets.get_path("naturalearth_cities"))
asia_cities = cities[cities.within(asia.unary_union)]
asia_cities.plot(ax=ax, color="red", markersize=15, marker="*", label="城市")

ax.set_title("亚洲人口分布与主要城市", fontsize=14)
ax.legend(loc="lower left")
ax.set_axis_off()
fig.savefig(output_dir / "asia_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("已保存: asia_overlay.png")

print("\n所有静态地图已保存至 output 目录")
