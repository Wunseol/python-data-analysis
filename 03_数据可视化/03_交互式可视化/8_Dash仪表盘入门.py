# 数据来源: 自造模拟数据
# Dash 是 Plotly 官方的 Python Web 应用框架，用于构建数据仪表盘
# 核心概念: Layout (布局) + Callbacks (回调交互)
# 依赖: pip install dash

import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

print("=" * 60)
print("Dash 仪表盘入门")
print("=" * 60)

# ============================================================
# 1. 准备数据
# ============================================================
print("\n--- 1. 准备模拟数据 ---")

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
    "销售额": np.cumsum(np.random.randn(n) * 10 + 50),
    "利润": np.cumsum(np.random.randn(n) * 5 + 20),
    "订单量": np.random.randint(10, 200, n),
    "类别": np.random.choice(["电子产品", "服装", "食品", "家居"], n),
    "地区": np.random.choice(["华东", "华南", "华北", "西南"], n),
})

df["月份"] = df["日期"].dt.to_period("M").astype(str)
df["月份简写"] = df["日期"].dt.strftime("%Y-%m")

print(f"数据集形状: {df.shape}")
print(df.head())

# ============================================================
# 2. 创建 Dash 应用
# ============================================================
print("\n--- 2. 创建 Dash 应用 ---")

app = dash.Dash(__name__)

# ============================================================
# 3. 定义 Layout (布局)
# ============================================================
print("\n--- 3. 定义 Layout ---")

app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "padding": "20px"}, children=[

    html.H1("📊 销售数据仪表盘", style={"textAlign": "center", "color": "#2C3E50"}),

    html.P("基于 Dash 构建的交互式数据报告", style={"textAlign": "center", "color": "#7F8C8D"}),

    html.Hr(),

    html.Div([
        html.Div([
            html.H3("选择地区:"),
            dcc.Dropdown(
                id="region-dropdown",
                options=[{"label": r, "value": r} for r in df["地区"].unique()],
                value="华东",
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"display": "inline-block", "marginRight": "40px"}),

        html.Div([
            html.H3("选择类别:"),
            dcc.Dropdown(
                id="category-dropdown",
                options=[{"label": "全部", "value": "全部"}] +
                        [{"label": c, "value": c} for c in df["类别"].unique()],
                value="全部",
                clearable=False,
                style={"width": "200px"},
            ),
        ], style={"display": "inline-block"}),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.Div([
            html.H4(id="kpi-sales", style={"textAlign": "center", "color": "#2980B9"}),
        ], style={"width": "23%", "display": "inline-block", "backgroundColor": "#EBF5FB",
                   "padding": "15px", "borderRadius": "10px", "marginRight": "2%"}),

        html.Div([
            html.H4(id="kpi-profit", style={"textAlign": "center", "color": "#27AE60"}),
        ], style={"width": "23%", "display": "inline-block", "backgroundColor": "#EAFAF1",
                   "padding": "15px", "borderRadius": "10px", "marginRight": "2%"}),

        html.Div([
            html.H4(id="kpi-orders", style={"textAlign": "center", "color": "#E67E22"}),
        ], style={"width": "23%", "display": "inline-block", "backgroundColor": "#FEF9E7",
                   "padding": "15px", "borderRadius": "10px", "marginRight": "2%"}),

        html.Div([
            html.H4(id="kpi-avg", style={"textAlign": "center", "color": "#8E44AD"}),
        ], style={"width": "23%", "display": "inline-block", "backgroundColor": "#F5EEF8",
                   "padding": "15px", "borderRadius": "10px"}),
    ], style={"marginBottom": "20px"}),

    html.Div([
        dcc.Graph(id="sales-trend"),
    ], style={"width": "48%", "display": "inline-block"}),

    html.Div([
        dcc.Graph(id="category-pie"),
    ], style={"width": "48%", "display": "inline-block", "marginLeft": "2%"}),

    html.Div([
        dcc.Graph(id="region-bar"),
    ], style={"width": "48%", "display": "inline-block", "marginTop": "20px"}),

    html.Div([
        dcc.Graph(id="scatter-plot"),
    ], style={"width": "48%", "display": "inline-block", "marginLeft": "2%", "marginTop": "20px"}),

])

# ============================================================
# 4. 定义 Callbacks (回调)
# ============================================================
print("\n--- 4. 定义 Callbacks ---")

@app.callback(
    [
        Output("kpi-sales", "children"),
        Output("kpi-profit", "children"),
        Output("kpi-orders", "children"),
        Output("kpi-avg", "children"),
        Output("sales-trend", "figure"),
        Output("category-pie", "figure"),
        Output("region-bar", "figure"),
        Output("scatter-plot", "figure"),
    ],
    [
        Input("region-dropdown", "value"),
        Input("category-dropdown", "value"),
    ],
)
def update_dashboard(selected_region, selected_category):
    dff = df[df["地区"] == selected_region]
    if selected_category != "全部":
        dff = dff[dff["类别"] == selected_category]

    total_sales = dff["销售额"].sum()
    total_profit = dff["利润"].sum()
    total_orders = dff["订单量"].sum()
    avg_order = total_sales / max(total_orders, 1)

    kpi_sales = f"总销售额\n¥{total_sales:,.0f}"
    kpi_profit = f"总利润\n¥{total_profit:,.0f}"
    kpi_orders = f"总订单量\n{total_orders:,}"
    kpi_avg = f"客单价\n¥{avg_order:,.1f}"

    fig_trend = px.line(
        dff, x="日期", y="销售额", title="销售趋势",
        template="plotly_white",
    )
    fig_trend.update_layout(margin=dict(l=40, r=20, t=40, b=30))

    cat_data = dff.groupby("类别")["销售额"].sum().reset_index()
    fig_pie = px.pie(
        cat_data, values="销售额", names="类别", title="类别占比",
        hole=0.4,
    )
    fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=30))

    region_data = dff.groupby("地区")["销售额"].sum().reset_index()
    fig_bar = px.bar(
        region_data, x="地区", y="销售额", title="地区销售额",
        template="plotly_white", color="地区",
    )
    fig_bar.update_layout(margin=dict(l=40, r=20, t=40, b=30), showlegend=False)

    fig_scatter = px.scatter(
        dff, x="销售额", y="利润", color="类别", size="订单量",
        title="销售额 vs 利润", template="plotly_white",
    )
    fig_scatter.update_layout(margin=dict(l=40, r=20, t=40, b=30))

    return kpi_sales, kpi_profit, kpi_orders, kpi_avg, fig_trend, fig_pie, fig_bar, fig_scatter

# ============================================================
# 5. 运行应用
# ============================================================
print("\n--- 5. 运行应用 ---")
print("""
Dash 应用运行方式:
    app.run_server(debug=True)

运行后在浏览器访问: http://127.0.0.1:8050

核心概念总结:
1. dash.Dash(__name__)  - 创建应用实例
2. html.Div / html.H1   - HTML组件，构建页面结构
3. dcc.Graph            - 图表组件，嵌入Plotly图表
4. dcc.Dropdown         - 下拉选择组件
5. @app.callback        - 回调装饰器，实现交互逻辑
6. Input / Output       - 定义回调的输入输出
7. app.layout           - 定义应用布局
8. app.run_server()     - 启动开发服务器

回调工作流程:
    用户操作(选择下拉框) -> Input触发 -> callback函数执行 -> Output更新
""")

if __name__ == "__main__":
    print("\n启动 Dash 应用 (访问 http://127.0.0.1:8050)...")
    print("按 Ctrl+C 停止服务器")
    app.run_server(debug=True, port=8050)
