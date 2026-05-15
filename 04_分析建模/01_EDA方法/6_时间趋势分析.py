# 数据来源: 自构建时间序列数据
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 6. 时间趋势分析
# 时间序列分析用于发现数据随时间变化的规律和趋势
# ============================================================

np.random.seed(42)
print("=" * 60)
print("时间趋势分析 - 自构建销售数据")
print("=" * 60)

# --------------------------------------------------
# 6.1 创建 DatetimeIndex
# 时间序列分析的基础是正确的时间索引
# --------------------------------------------------
date_range = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
n = len(date_range)

trend = np.linspace(100, 200, n)
seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.normal(0, 8, n)
sales = trend + seasonality + noise

df = pd.DataFrame({
    'date': date_range,
    'sales': sales.round(2),
    'visitors': (sales * np.random.uniform(2, 5, n)).astype(int),
    'category': np.random.choice(['电子产品', '服装', '食品', '家居'], n),
})
df.set_index('date', inplace=True)

print("\n【6.1 时间序列数据概览】")
print(df.head(10))
print(f"\n索引类型: {type(df.index)}")
print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
print(f"数据频率: {pd.infer_freq(df.index)}")

# --------------------------------------------------
# 6.2 折线图 - 时间趋势可视化
# 最基本的时间序列可视化方法
# --------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

axes[0, 0].plot(df.index, df['sales'], linewidth=0.5, color='steelblue', alpha=0.7)
axes[0, 0].set_title('每日销售额趋势')
axes[0, 0].set_xlabel('日期')
axes[0, 0].set_ylabel('销售额')
axes[0, 0].grid(True, alpha=0.3)

# --------------------------------------------------
# 6.3 移动平均 (Rolling Mean)
# 平滑短期波动,展示长期趋势
# --------------------------------------------------
df['sales_ma7'] = df['sales'].rolling(window=7).mean()
df['sales_ma30'] = df['sales'].rolling(window=30).mean()
df['sales_ma90'] = df['sales'].rolling(window=90).mean()

axes[0, 1].plot(df.index, df['sales'], linewidth=0.3, alpha=0.3, color='gray', label='原始数据')
axes[0, 1].plot(df.index, df['sales_ma7'], linewidth=1, color='orange', label='7日均线')
axes[0, 1].plot(df.index, df['sales_ma30'], linewidth=1.5, color='green', label='30日均线')
axes[0, 1].plot(df.index, df['sales_ma90'], linewidth=2, color='red', label='90日均线')
axes[0, 1].set_title('移动平均趋势')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --------------------------------------------------
# 6.4 resample() - 重采样
# 将高频数据聚合为低频数据,便于观察不同时间尺度的趋势
# --------------------------------------------------
monthly_sales = df['sales'].resample('M').agg(['mean', 'sum', 'min', 'max'])
print("\n【6.4 月度重采样统计】")
print(monthly_sales.head(12).round(2))

axes[1, 0].plot(monthly_sales.index, monthly_sales['mean'], marker='o',
                markersize=3, linewidth=1.5, color='teal', label='月均值')
axes[1, 0].fill_between(monthly_sales.index, monthly_sales['min'],
                         monthly_sales['max'], alpha=0.2, color='teal')
axes[1, 0].set_title('月度销售额(均值+范围)')
axes[1, 0].set_xlabel('月份')
axes[1, 0].set_ylabel('销售额')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

quarterly_sales = df['sales'].resample('Q').mean()
axes[1, 1].bar(quarterly_sales.index.strftime('%Y-Q%q'), quarterly_sales.values,
               color='coral', edgecolor='white')
axes[1, 1].set_title('季度平均销售额')
axes[1, 1].set_xlabel('季度')
axes[1, 1].set_ylabel('平均销售额')
axes[1, 1].tick_params(axis='x', rotation=45)

# --------------------------------------------------
# 6.5 pct_change() - 变化率分析
# 计算环比变化率,衡量增长速度
# --------------------------------------------------
df['sales_pct_change'] = df['sales'].pct_change()
monthly_pct = df['sales'].resample('M').mean().pct_change()

print("\n【6.5 月度环比增长率】")
print(monthly_pct.head(12).round(4))

axes[2, 0].bar(monthly_pct.index[1:], monthly_pct.values[1:] * 100,
               color=['green' if x > 0 else 'red' for x in monthly_pct.values[1:]],
               alpha=0.7, width=20)
axes[2, 0].axhline(y=0, color='black', linewidth=0.5)
axes[2, 0].set_title('月度环比增长率(%)')
axes[2, 0].set_xlabel('月份')
axes[2, 0].set_ylabel('增长率(%)')
axes[2, 0].grid(True, alpha=0.3)

# --------------------------------------------------
# 6.6 趋势可视化 - 年度对比
# --------------------------------------------------
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_year'] = df.index.dayofyear

for year in df['year'].unique():
    yearly_data = df[df['year'] == year]
    monthly_avg = yearly_data.groupby('month')['sales'].mean()
    axes[2, 1].plot(monthly_avg.index, monthly_avg.values, marker='o',
                    markersize=4, label=str(year))

axes[2, 1].set_title('年度销售额对比(按月)')
axes[2, 1].set_xlabel('月份')
axes[2, 1].set_ylabel('月均销售额')
axes[2, 1].set_xticks(range(1, 13))
axes[2, 1].set_xticklabels([f'{m}月' for m in range(1, 13)])
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / '6_时间趋势分析.png', dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------
# 补充: 时间序列分解概念
# --------------------------------------------------
print("\n【补充: 年度统计汇总】")
yearly_summary = df.groupby('year')['sales'].agg(['mean', 'std', 'min', 'max'])
print(yearly_summary.round(2))

print(f"\n整体年增长率: {(yearly_summary.loc[2024, 'mean'] / yearly_summary.loc[2022, 'mean'] - 1) * 100:.2f}%")

print(f"\n图表已保存至: {output_dir}")
print("\n" + "=" * 60)
print("时间趋势分析完成! 移动平均和重采样是时间序列分析的核心方法。")
print("=" * 60)
