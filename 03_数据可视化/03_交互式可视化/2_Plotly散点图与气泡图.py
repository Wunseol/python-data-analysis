# 数据来源: Plotly内置 gapminder 数据集
# gapminder 数据集包含全球各国家的人口、GDP、预期寿命等指标
# 散点图和气泡图是探索多变量关系的利器

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

print("=" * 60)
print("Plotly 散点图与气泡图")
print("=" * 60)

# ============================================================
# 1. 加载 gapminder 数据集
# ============================================================
print("\n--- 1. gapminder 数据集 ---")

df = px.data.gapminder()
print(f"数据集形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"国家数量: {df['country'].nunique()}")
print(df.head(10))

# ============================================================
# 2. 基础散点图 - px.scatter()
# ============================================================
print("\n--- 2. 基础散点图 ---")

df_2007 = df[df["year"] == 2007]

fig_basic = px.scatter(
    df_2007,
    x="gdpPercap",
    y="lifeExp",
    title="2007年各国人均GDP与预期寿命",
    labels={"gdpPercap": "人均GDP (美元)", "lifeExp": "预期寿命 (岁)"},
)

fig_basic.update_layout(template="plotly_white")

fig_basic.show()
fig_basic.write_html("output_散点图_基础.html")
print("基础散点图已保存")

# ============================================================
# 3. 带颜色映射的散点图 - color 参数
# ============================================================
print("\n--- 3. 带颜色映射的散点图 ---")

fig_color = px.scatter(
    df_2007,
    x="gdpPercap",
    y="lifeExp",
    color="continent",
    title="2007年各大洲人均GDP与预期寿命",
    labels={
        "gdpPercap": "人均GDP (美元)",
        "lifeExp": "预期寿命 (岁)",
        "continent": "大洲",
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
)

fig_color.update_layout(template="plotly_white")

fig_color.show()
fig_color.write_html("output_散点图_颜色映射.html")
print("颜色映射散点图已保存")

# ============================================================
# 4. 气泡图 - size 参数
# ============================================================
print("\n--- 4. 气泡图 (size参数) ---")

fig_bubble = px.scatter(
    df_2007,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    size_max=60,
    title="2007年全球气泡图: GDP vs 预期寿命 (气泡大小=人口)",
    labels={
        "gdpPercap": "人均GDP (美元)",
        "lifeExp": "预期寿命 (岁)",
        "continent": "大洲",
        "pop": "人口",
    },
)

fig_bubble.update_layout(template="plotly_white")

fig_bubble.show()
fig_bubble.write_html("output_气泡图.html")
print("气泡图已保存")

# ============================================================
# 5. hover_name 与 hover_data 自定义悬停信息
# ============================================================
print("\n--- 5. 自定义悬停信息 ---")

fig_hover = px.scatter(
    df_2007,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    hover_data={
        "pop": True,
        "gdpPercap": ":.2f",
        "lifeExp": ":.1f",
        "continent": False,
    },
    size_max=50,
    title="自定义悬停信息 - 鼠标悬停查看详细数据",
)

fig_hover.update_layout(template="plotly_white")

fig_hover.show()
fig_hover.write_html("output_散点图_悬停信息.html")
print("自定义悬停信息散点图已保存")

# ============================================================
# 6. facet_col 分面散点图
# ============================================================
print("\n--- 6. facet_col 分面散点图 ---")

fig_facet = px.scatter(
    df[df["year"].isin([1952, 1977, 2002])],
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    facet_col="year",
    hover_name="country",
    size_max=50,
    title="不同年份全球GDP与预期寿命对比",
    labels={"gdpPercap": "人均GDP", "lifeExp": "预期寿命", "continent": "大洲"},
)

fig_facet.update_layout(template="plotly_white")

fig_facet.show()
fig_facet.write_html("output_散点图_分面.html")
print("分面散点图已保存")

# ============================================================
# 7. 对数坐标轴气泡图
# ============================================================
print("\n--- 7. 对数坐标轴气泡图 ---")

fig_log = px.scatter(
    df_2007,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
    title="对数坐标轴气泡图 (X轴取对数)",
)

fig_log.update_layout(template="plotly_white")

fig_log.show()
fig_log.write_html("output_气泡图_对数轴.html")
print("对数坐标轴气泡图已保存")

# ============================================================
# 8. 使用 graph_objects 手动创建气泡图
# ============================================================
print("\n--- 8. graph_objects 手动创建气泡图 ---")

continents = df_2007["continent"].unique()
colors = {"Asia": "#FF6B6B", "Europe": "#4ECDC4", "Africa": "#45B7D1",
          "Americas": "#96CEB4", "Oceania": "#FFEAA7"}

fig_go = go.Figure()

for continent in continents:
    df_c = df_2007[df_2007["continent"] == continent]
    fig_go.add_trace(go.Scatter(
        x=df_c["gdpPercap"],
        y=df_c["lifeExp"],
        mode="markers",
        name=continent,
        marker=dict(
            size=df_c["pop"] / df_c["pop"].max() * 60 + 5,
            color=colors.get(continent, "gray"),
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        text=df_c["country"],
        hovertemplate="<b>%{text}</b><br>GDP: %{x:,.0f}<br>寿命: %{y:.1f}<extra></extra>",
    ))

fig_go.update_layout(
    title="graph_objects 手动创建气泡图",
    xaxis_title="人均GDP (美元)",
    yaxis_title="预期寿命 (岁)",
    template="plotly_white",
)

fig_go.show()
fig_go.write_html("output_气泡图_go.html")
print("graph_objects气泡图已保存")

print("\n" + "=" * 60)
print("Plotly 散点图与气泡图演示完成！")
print("=" * 60)
