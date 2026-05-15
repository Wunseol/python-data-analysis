# 依赖库最低版本要求: plotly>=5.0, pyecharts>=2.0, bokeh>=3.0, dash>=2.0
# 数据来源: Plotly内置数据集 (px.data) + 自造模拟数据
# Plotly Express 是 Plotly 的高级接口，可以用极少的代码创建交互式图表
# 交互特性: 鼠标悬停显示数据、缩放、平移、框选、导出图片等

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

print("=" * 60)
print("Plotly 基础交互图表")
print("=" * 60)

# ============================================================
# 1. 折线图 - px.line()
# ============================================================
print("\n--- 1. 折线图 px.line() ---")

df_stocks = px.data.stocks()
print(f"股票数据集形状: {df_stocks.shape}")
print(df_stocks.head())

fig_line = px.line(
    df_stocks,
    x="date",
    y=["GOOG", "AAPL", "AMZN", "FB", "NFLX", "MSFT"],
    title="科技公司股票价格走势",
    labels={"value": "标准化价格", "variable": "公司", "date": "日期"},
)

fig_line.update_layout(
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig_line.show()
fig_line.write_html("output_1_折线图.html")
print("折线图已保存为 output_1_折线图.html")

# ============================================================
# 2. 柱状图 - px.bar()
# ============================================================
print("\n--- 2. 柱状图 px.bar() ---")

df_tips = px.data.tips()
print(f"小费数据集形状: {df_tips.shape}")
print(df_tips.head())

fig_bar = px.bar(
    df_tips,
    x="day",
    y="total_bill",
    color="sex",
    barmode="group",
    title="各天不同性别的消费总额",
    labels={"total_bill": "消费总额 ($)", "day": "星期", "sex": "性别"},
)

fig_bar.update_layout(template="plotly_white")

fig_bar.show()
fig_bar.write_html("output_2_柱状图.html")
print("柱状图已保存为 output_2_柱状图.html")

# ============================================================
# 3. 散点图 - px.scatter()
# ============================================================
print("\n--- 3. 散点图 px.scatter() ---")

fig_scatter = px.scatter(
    df_tips,
    x="total_bill",
    y="tip",
    color="time",
    size="size",
    title="消费金额与小费的关系",
    labels={"total_bill": "消费金额 ($)", "tip": "小费 ($)", "time": "用餐时间"},
)

fig_scatter.update_layout(template="plotly_white")

fig_scatter.show()
fig_scatter.write_html("output_3_散点图.html")
print("散点图已保存为 output_3_散点图.html")

# ============================================================
# 4. 直方图 - px.histogram()
# ============================================================
print("\n--- 4. 直方图 px.histogram() ---")

fig_hist = px.histogram(
    df_tips,
    x="total_bill",
    nbins=30,
    color="sex",
    marginal="box",
    title="消费金额分布直方图",
    labels={"total_bill": "消费金额 ($)", "sex": "性别"},
)

fig_hist.update_layout(template="plotly_white", bargap=0.1)

fig_hist.show()
fig_hist.write_html("output_4_直方图.html")
print("直方图已保存为 output_4_直方图.html")

# ============================================================
# 5. update_layout() 自定义布局
# ============================================================
print("\n--- 5. update_layout() 自定义布局 ---")

np.random.seed(42)
df_custom = pd.DataFrame({
    "月份": pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m"),
    "销售额": np.random.randint(100, 500, 12),
    "利润": np.random.randint(20, 150, 12),
})

fig_custom = px.bar(
    df_custom,
    x="月份",
    y="销售额",
    title="2024年月度销售额",
)

fig_custom.update_layout(
    title=dict(text="2024年月度销售额", font=dict(size=20, color="navy"), x=0.5),
    xaxis=dict(title="月份", tickangle=45),
    yaxis=dict(title="销售额 (万元)", gridcolor="lightgray"),
    plot_bgcolor="white",
    paper_bgcolor="lightyellow",
    font=dict(family="Arial", size=12),
    width=900,
    height=500,
)

fig_custom.update_traces(marker_color="steelblue", marker_line_color="navy", marker_line_width=1)

fig_custom.show()
fig_custom.write_html("output_5_自定义布局.html")
print("自定义布局图表已保存为 output_5_自定义布局.html")

# ============================================================
# 6. 使用 graph_objects 创建更精细的图表
# ============================================================
print("\n--- 6. graph_objects 精细控制 ---")

fig_go = go.Figure()

fig_go.add_trace(go.Scatter(
    x=df_custom["月份"],
    y=df_custom["销售额"],
    mode="lines+markers",
    name="销售额",
    line=dict(color="royalblue", width=2),
    marker=dict(size=8),
))

fig_go.add_trace(go.Scatter(
    x=df_custom["月份"],
    y=df_custom["利润"],
    mode="lines+markers",
    name="利润",
    line=dict(color="firebrick", width=2, dash="dash"),
    marker=dict(size=8, symbol="diamond"),
))

fig_go.update_layout(
    title="销售额与利润趋势对比",
    xaxis_title="月份",
    yaxis_title="金额 (万元)",
    template="plotly_white",
    hovermode="x",
)

fig_go.show()
fig_go.write_html("output_6_graph_objects.html")
print("graph_objects图表已保存为 output_6_graph_objects.html")

print("\n" + "=" * 60)
print("Plotly 基础交互图表演示完成！")
print("所有HTML文件已保存到当前目录，可在浏览器中打开查看交互效果。")
print("=" * 60)
