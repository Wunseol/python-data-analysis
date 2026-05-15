# 数据来源: 自建模拟数据

import folium
import pandas as pd
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# === 1. 创建基础地图 ===
m = folium.Map(location=[35.86, 104.19], zoom_start=4, tiles="OpenStreetMap")
m.save(output_dir / "basic_map.html")
print("已保存: basic_map.html")

# === 2. 添加标记Marker ===
cities = pd.DataFrame({
    "城市": ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉"],
    "纬度": [39.9, 31.2, 23.1, 22.5, 30.6, 30.3, 30.6],
    "经度": [116.4, 121.5, 113.3, 114.1, 104.1, 120.2, 114.3],
    "人口(万)": [2189, 2487, 1868, 1756, 2094, 1237, 1365],
    "类型": ["直辖市", "直辖市", "省会", "经济特区", "省会", "省会", "省会"],
})

m = folium.Map(location=[33, 112], zoom_start=5)

for _, row in cities.iterrows():
    folium.Marker(
        location=[row["纬度"], row["经度"]],
        popup=folium.Popup(
            f"<b>{row['城市']}</b><br>人口: {row['人口(万)']}万<br>类型: {row['类型']}",
            max_width=200,
        ),
        tooltip=row["城市"],
        icon=folium.Icon(color="red" if row["类型"] == "直辖市" else "blue",
                         icon="info-sign"),
    ).add_to(m)

m.save(output_dir / "markers_map.html")
print("已保存: markers_map.html")

# === 3. 圆形标记CircleMarker ===
m = folium.Map(location=[33, 112], zoom_start=5)

for _, row in cities.iterrows():
    folium.CircleMarker(
        location=[row["纬度"], row["经度"]],
        radius=row["人口(万)"] / 300,
        popup=f"{row['城市']}: {row['人口(万)']}万",
        color="crimson",
        fill=True,
        fill_color="crimson",
        fill_opacity=0.6,
    ).add_to(m)

m.save(output_dir / "circle_markers_map.html")
print("已保存: circle_markers_map.html")

# === 4. 热力图HeatMap ===
from folium.plugins import HeatMap

m = folium.Map(location=[33, 112], zoom_start=5)

heat_data = [
    [row["纬度"], row["经度"], row["人口(万)"]]
    for _, row in cities.iterrows()
]
HeatMap(
    heat_data,
    min_opacity=0.3,
    max_zoom=10,
    radius=40,
    blur=30,
    gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
).add_to(m)

m.save(output_dir / "heatmap.html")
print("已保存: heatmap.html")

# === 5. 图层控制 ===
m = folium.Map(location=[33, 112], zoom_start=5)

marker_group = folium.FeatureGroup(name="城市标记")
heatmap_group = folium.FeatureGroup(name="人口热力图")

for _, row in cities.iterrows():
    folium.Marker(
        location=[row["纬度"], row["经度"]],
        popup=f"{row['城市']}: {row['人口(万)']}万",
        tooltip=row["城市"],
    ).add_to(marker_group)

HeatMap(heat_data, radius=40, blur=30).add_to(heatmap_group)

marker_group.add_to(m)
heatmap_group.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

m.save(output_dir / "layer_control_map.html")
print("已保存: layer_control_map.html")

# === 6. 多种底图切换 ===
m = folium.Map(location=[33, 112], zoom_start=5)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer("Stamen Terrain", name="地形图").add_to(m)
folium.TileLayer("CartoDB positron", name="浅色底图").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="深色底图").add_to(m)

for _, row in cities.iterrows():
    folium.Marker(
        location=[row["纬度"], row["经度"]],
        popup=row["城市"],
    ).add_to(m)

folium.LayerControl().add_to(m)
m.save(output_dir / "tile_layers_map.html")
print("已保存: tile_layers_map.html")

print("\n所有交互式地图已保存至 output 目录")
