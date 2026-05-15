# 数据来源: seaborn内置 titanic 数据集 + Plotly内置数据集
# 综合案例: 使用 Plotly 创建多图表交互式数据报告
# 涵盖: 子图布局、多图表类型组合、下拉筛选概念、HTML导出

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
import numpy as np

print("=" * 60)
print("综合案例: 交互式数据报告")
print("=" * 60)

# ============================================================
# 1. 加载 Titanic 数据集
# ============================================================
print("\n--- 1. 加载数据 ---")

df_titanic = sns.load_dataset("titanic")
print(f"Titanic 数据集形状: {df_titanic.shape}")
print(f"列名: {list(df_titanic.columns)}")
print(df_titanic.head())

df_titanic["存活"] = df_titanic["survived"].map({0: "未存活", 1: "存活"})
df_titanic["舱位"] = df_titanic["pclass"].map({1: "一等舱", 2: "二等舱", 3: "三等舱"})
df_titanic["性别"] = df_titanic["sex"].map({"male": "男性", "female": "女性"})

# ============================================================
# 2. 单图表: 存活率分析
# ============================================================
print("\n--- 2. 存活率分析 ---")

survival_by_class = df_titanic.groupby(["舱位", "存活"]).size().reset_index(name="人数")

fig_survival = px.bar(
    survival_by_class,
    x="舱位",
    y="人数",
    color="存活",
    barmode="group",
    title="各舱位存活情况",
    color_discrete_map={"存活": "#2ECC71", "未存活": "#E74C3C"},
    category_orders={"舱位": ["一等舱", "二等舱", "三等舱"]},
)

fig_survival.update_layout(template="plotly_white")

fig_survival.show()
fig_survival.write_html("output_报告_存活率.html")
print("存活率分析图已保存")

# ============================================================
# 3. 单图表: 年龄分布
# ============================================================
print("\n--- 3. 年龄分布 ---")

fig_age = px.histogram(
    df_titanic.dropna(subset=["age"]),
    x="age",
    color="存活",
    nbins=30,
    marginal="violin",
    title="乘客年龄分布",
    labels={"age": "年龄", "存活": "状态"},
    color_discrete_map={"存活": "#2ECC71", "未存活": "#E74C3C"},
)

fig_age.update_layout(template="plotly_white", bargap=0.05)

fig_age.show()
fig_age.write_html("output_报告_年龄分布.html")
print("年龄分布图已保存")

# ============================================================
# 4. 单图表: 票价与年龄散点图
# ============================================================
print("\n--- 4. 票价与年龄散点图 ---")

fig_scatter = px.scatter(
    df_titanic.dropna(subset=["age", "fare"]),
    x="age",
    y="fare",
    color="存活",
    size="pclass",
    facet_col="性别",
    hover_name="舱位",
    title="票价 vs 年龄 (按存活状态着色)",
    labels={"age": "年龄", "fare": "票价 (£)", "存活": "状态"},
    color_discrete_map={"存活": "#2ECC71", "未存活": "#E74C3C"},
)

fig_scatter.update_layout(template="plotly_white")

fig_scatter.show()
fig_scatter.write_html("output_报告_散点图.html")
print("散点图已保存")

# ============================================================
# 5. 子图布局 - make_subplots
# ============================================================
print("\n--- 5. 子图布局 make_subplots ---")

fig_subplots = make_subplots(
    rows=2, cols=2,
    subplot_titles=("各舱位存活率", "性别存活率", "登船港口分布", "年龄箱线图"),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "pie"}, {"type": "box"}],
    ],
)

survival_rate_class = df_titanic.groupby("舱位")["survived"].mean().reset_index()
survival_rate_class.columns = ["舱位", "存活率"]
survival_rate_class["存活率"] = (survival_rate_class["存活率"] * 100).round(1)

fig_subplots.add_trace(
    go.Bar(
        x=survival_rate_class["舱位"],
        y=survival_rate_class["存活率"],
        marker_color=["#FFD700", "#C0C0C0", "#CD7F32"],
        name="存活率",
        showlegend=False,
    ),
    row=1, col=1,
)

survival_rate_sex = df_titanic.groupby("性别")["survived"].mean().reset_index()
survival_rate_sex["存活率"] = (survival_rate_sex["survived"] * 100).round(1)

fig_subplots.add_trace(
    go.Bar(
        x=survival_rate_sex["性别"],
        y=survival_rate_sex["存活率"],
        marker_color=["#3498DB", "#E91E63"],
        name="存活率",
        showlegend=False,
    ),
    row=1, col=2,
)

embarked_counts = df_titanic["embark_town"].value_counts().reset_index()
embarked_counts.columns = ["港口", "人数"]

fig_subplots.add_trace(
    go.Pie(
        labels=embarked_counts["港口"],
        values=embarked_counts["人数"],
        name="港口",
    ),
    row=2, col=1,
)

df_age_clean = df_titanic.dropna(subset=["age"])
for survived, color in [(0, "#E74C3C"), (1, "#2ECC71")]:
    df_sub = df_age_clean[df_age_clean["survived"] == survived]
    fig_subplots.add_trace(
        go.Box(
            y=df_sub["age"],
            name="存活" if survived else "未存活",
            marker_color=color,
        ),
        row=2, col=2,
    )

fig_subplots.update_layout(
    title_text="Titanic 数据综合分析报告",
    title_font_size=20,
    height=800,
    template="plotly_white",
)

fig_subplots.show()
fig_subplots.write_html("output_报告_子图布局.html")
print("子图布局报告已保存")

# ============================================================
# 6. 带下拉筛选的交互式报告
# ============================================================
print("\n--- 6. 带下拉筛选的交互式报告 ---")

classes = ["全部", "一等舱", "二等舱", "三等舱"]

fig_dropdown = go.Figure()

for cls in classes:
    if cls == "全部":
        dff = df_titanic
    else:
        dff = df_titanic[df_titanic["舱位"] == cls]

    age_hist = dff["age"].dropna()
    fig_dropdown.add_trace(
        go.Histogram(
            x=age_hist,
            name=cls,
            visible=(cls == "全部"),
            nbinsx=30,
            marker_color={"全部": "#3498DB", "一等舱": "#FFD700",
                          "二等舱": "#C0C0C0", "三等舱": "#CD7F32"}[cls],
        )
    )

dropdown_buttons = []
for i, cls in enumerate(classes):
    visibility = [j == i for j in range(len(classes))]
    if cls == "全部":
        count = len(df_titanic)
    else:
        count = len(df_titanic[df_titanic["舱位"] == cls])
    dropdown_buttons.append(dict(
        label=f"{cls} ({count}人)",
        method="update",
        args=[{"visible": visibility}, {"title": f"年龄分布 - {cls}"}],
    ))

fig_dropdown.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=dropdown_buttons,
        direction="down",
        x=0.5,
        xanchor="center",
        y=1.15,
        yanchor="top",
    )],
    title="年龄分布 - 全部",
    xaxis_title="年龄",
    yaxis_title="人数",
    template="plotly_white",
    bargap=0.05,
)

fig_dropdown.show()
fig_dropdown.write_html("output_报告_下拉筛选.html")
print("下拉筛选报告已保存")

# ============================================================
# 7. 综合仪表盘式报告
# ============================================================
print("\n--- 7. 综合仪表盘式报告 ---")

fig_dashboard = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        "存活率总览", "各舱位存活率", "性别存活率",
        "年龄分布", "票价分布", "兄弟姐妹/配偶数",
        "各港口登船人数", "舱位与票价", "年龄vs票价",
    ),
    specs=[
        [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
        [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
        [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.1,
)

survived_counts = df_titanic["存活"].value_counts()
fig_dashboard.add_trace(go.Pie(labels=survived_counts.index, values=survived_counts.values,
                                marker_colors=["#E74C3C", "#2ECC71"]), row=1, col=1)

fig_dashboard.add_trace(go.Bar(x=survival_rate_class["舱位"], y=survival_rate_class["存活率"],
                                marker_color=["#FFD700", "#C0C0C0", "#CD7F32"], showlegend=False), row=1, col=2)

fig_dashboard.add_trace(go.Bar(x=survival_rate_sex["性别"], y=survival_rate_sex["存活率"],
                                marker_color=["#3498DB", "#E91E63"], showlegend=False), row=1, col=3)

fig_dashboard.add_trace(go.Histogram(x=df_titanic["age"].dropna(), marker_color="#3498DB",
                                      showlegend=False, nbinsx=30), row=2, col=1)

fig_dashboard.add_trace(go.Histogram(x=df_titanic["fare"].dropna(), marker_color="#E67E22",
                                      showlegend=False, nbinsx=30), row=2, col=2)

fig_dashboard.add_trace(go.Histogram(x=df_titanic["sibsp"], marker_color="#9B59B6",
                                      showlegend=False), row=2, col=3)

embarked_counts = df_titanic["embark_town"].value_counts().reset_index()
embarked_counts.columns = ["港口", "人数"]
fig_dashboard.add_trace(go.Bar(x=embarked_counts["港口"], y=embarked_counts["人数"],
                                marker_color="#1ABC9C", showlegend=False), row=3, col=1)

for pclass, color in [(1, "#FFD700"), (2, "#C0C0C0"), (3, "#CD7F32")]:
    df_pc = df_titanic[df_titanic["pclass"] == pclass]["fare"].dropna()
    fig_dashboard.add_trace(go.Box(y=df_pc, name=f"{pclass}等舱",
                                    marker_color=color), row=3, col=2)

df_clean = df_titanic.dropna(subset=["age", "fare"])
fig_dashboard.add_trace(go.Scatter(
    x=df_clean["age"], y=df_clean["fare"],
    mode="markers", marker=dict(
        color=df_clean["survived"], colorscale=["#E74C3C", "#2ECC71"],
        size=4, opacity=0.5,
    ),
    showlegend=False,
), row=3, col=3)

fig_dashboard.update_layout(
    title_text="Titanic 数据综合仪表盘",
    title_font_size=22,
    height=1200,
    template="plotly_white",
    showlegend=True,
)

fig_dashboard.show()
fig_dashboard.write_html("output_报告_综合仪表盘.html")
print("综合仪表盘报告已保存")

# ============================================================
# 8. 导出完整报告说明
# ============================================================
print("\n--- 8. 导出说明 ---")
print("""
交互式报告导出方式:
1. fig.write_html("report.html")
   - 导出为独立HTML文件，包含完整交互功能
   - 可直接在浏览器中打开，无需服务器
   - 可嵌入到网页中

2. fig.write_html("report.html", include_plotlyjs="cdn")
   - 使用CDN加载Plotly.js，文件更小
   - 需要网络连接才能正常显示

3. fig.write_image("report.png")
   - 导出为静态图片 (需要 kaleido 库)
   - pip install kaleido

4. 多图表合并导出:
   - 使用 make_subplots 创建子图布局
   - 或使用下拉菜单/选项卡切换不同视图
   - 单个HTML文件包含所有交互
""")

print("\n" + "=" * 60)
print("综合案例: 交互式数据报告演示完成！")
print("所有HTML文件已保存到当前目录，可在浏览器中打开查看交互效果。")
print("生成的文件列表:")
print("  - output_报告_存活率.html")
print("  - output_报告_年龄分布.html")
print("  - output_报告_散点图.html")
print("  - output_报告_子图布局.html")
print("  - output_报告_下拉筛选.html")
print("  - output_报告_综合仪表盘.html")
print("=" * 60)
