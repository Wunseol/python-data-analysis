# 数据来源: Plotly内置 gapminder 数据集
# Plotly 动画图表可以让数据"动起来"，展示随时间变化的趋势
# 核心参数: animation_frame (时间维度), animation_group (实体追踪)

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

print("=" * 60)
print("Plotly 动态动画图表")
print("=" * 60)

# ============================================================
# 1. gapminder 动画散点图 - animation_frame
# ============================================================
print("\n--- 1. gapminder 动画散点图 ---")

df = px.data.gapminder()

fig_anim = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    animation_frame="year",
    animation_group="country",
    log_x=True,
    size_max=55,
    range_x=[100, 100000],
    range_y=[25, 90],
    title="全球发展动画: GDP vs 预期寿命 (1952-2007)",
    labels={
        "gdpPercap": "人均GDP (对数)",
        "lifeExp": "预期寿命 (岁)",
        "continent": "大洲",
        "pop": "人口",
    },
    color_discrete_map={
        "Asia": "#FF6B6B",
        "Europe": "#4ECDC4",
        "Africa": "#45B7D1",
        "Americas": "#96CEB4",
        "Oceania": "#FFEAA7",
    },
)

fig_anim.update_layout(template="plotly_white")

fig_anim.show()
fig_anim.write_html("output_动画_gapminder.html")
print("gapminder动画散点图已保存")

# ============================================================
# 2. animation_group 的作用说明
# ============================================================
print("\n--- 2. animation_group 参数说明 ---")
print("""
animation_group 的作用:
- 指定数据点的唯一标识，确保动画播放时同一实体被正确追踪
- 如果不设置 animation_group，动画帧之间无法对应同一数据点
- 通常设置为实体ID（如国家名、城市名等）
- 例如: animation_group="country" 确保每个国家在各帧间平滑过渡
""")

# ============================================================
# 3. range_x / range_y 固定坐标轴范围
# ============================================================
print("\n--- 3. 固定坐标轴范围 ---")

fig_fixed = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    animation_frame="year",
    log_x=True,
    size_max=55,
    range_x=[100, 100000],
    range_y=[25, 90],
    title="固定坐标轴范围 - 防止动画播放时坐标轴跳动",
)

fig_fixed.update_layout(template="plotly_white")

fig_fixed.show()
fig_fixed.write_html("output_动画_固定坐标轴.html")
print("固定坐标轴动画图已保存")

# 不固定坐标轴的对比
fig_free = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    animation_frame="year",
    log_x=True,
    size_max=55,
    title="不固定坐标轴 - 坐标轴会随数据变化而跳动",
)

fig_free.update_layout(template="plotly_white")

fig_free.show()
fig_free.write_html("output_动画_自由坐标轴.html")
print("自由坐标轴动画图已保存 (对比用)")

# ============================================================
# 4. 动画柱状图
# ============================================================
print("\n--- 4. 动画柱状图 ---")

df_europe = df[df["continent"] == "Europe"]

fig_bar_anim = px.bar(
    df_europe,
    x="country",
    y="pop",
    color="country",
    animation_frame="year",
    range_y=[0, 90000000],
    title="欧洲各国人口变化动画 (1952-2007)",
    labels={"pop": "人口", "country": "国家"},
)

fig_bar_anim.update_layout(
    template="plotly_white",
    showlegend=False,
    xaxis_tickangle=-45,
)

fig_bar_anim.show()
fig_bar_anim.write_html("output_动画_柱状图.html")
print("动画柱状图已保存")

# ============================================================
# 5. 自定义动画 - 使用 go.Figure 和 frames
# ============================================================
print("\n--- 5. 自定义动画 (go.Figure + frames) ---")

np.random.seed(42)
n_points = 50
years = list(range(2000, 2021))

fig_custom = go.Figure()

initial_x = np.random.randn(n_points)
initial_y = np.random.randn(n_points)

fig_custom.add_trace(go.Scatter(
    x=initial_x,
    y=initial_y,
    mode="markers",
    marker=dict(size=10, color="steelblue", opacity=0.7),
))

frames = []
for i, year in enumerate(years):
    drift_x = initial_x + i * 0.05 * np.random.randn(n_points)
    drift_y = initial_y + i * 0.05 * np.random.randn(n_points)
    frames.append(go.Frame(
        data=[go.Scatter(x=drift_x, y=drift_y, mode="markers")],
        name=str(year),
        layout=go.Layout(title_text=f"粒子漂移动画 - {year}年"),
    ))

fig_custom.frames = frames

fig_custom.update_layout(
    title="粒子漂移动画 - 2000年",
    xaxis=dict(range=[-5, 5], autorange=False),
    yaxis=dict(range=[-5, 5], autorange=False),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(label="播放", method="animate", args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
            dict(label="暂停", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        active=0,
        steps=[dict(label=year, method="animate", args=[[str(year)], dict(mode="immediate", frame=dict(duration=200, redraw=True))]) for year in years],
        transition=dict(duration=0),
        x=0, y=0,
        currentvalue=dict(font=dict(size=12), prefix="年份: ", visible=True),
    )],
)

fig_custom.show()
fig_custom.write_html("output_动画_自定义frames.html")
print("自定义frames动画已保存")

# ============================================================
# 6. 动画速度与过渡效果控制
# ============================================================
print("\n--- 6. 动画控制说明 ---")
print("""
动画控制关键参数:
1. frame.duration - 每帧持续时间 (毫秒)，值越小越快
2. frame.redraw - 是否每帧重绘，建议True
3. transition.duration - 帧间过渡动画时间 (毫秒)
4. fromcurrent - 是否从当前帧继续播放
5. easing - 过渡缓动函数: linear, quad, cubic, elastic, bounce 等

在 update_layout 中设置:
    updatemenus=[dict(
        buttons=[
            dict(label="快进", method="animate",
                 args=[None, dict(frame=dict(duration=100))]),
            dict(label="慢放", method="animate",
                 args=[None, dict(frame=dict(duration=500))]),
        ]
    )]
""")

print("\n" + "=" * 60)
print("Plotly 动态动画图表演示完成！")
print("=" * 60)
