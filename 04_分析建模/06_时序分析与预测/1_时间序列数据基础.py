# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, statsmodels>=0.14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. pd.date_range() —— 生成日期范围
# ============================================================
print("=" * 60)
print("1. pd.date_range() 生成日期范围")
print("=" * 60)

# 按起止日期生成，频率为天
dates_daily = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print("每日频率 (freq='D'):")
print(dates_daily)

# 按起止日期生成，频率为月
dates_monthly = pd.date_range(start='2024-01-01', end='2024-12-31', freq='ME')
print("\n月末频率 (freq='ME'):")
print(dates_monthly)

# 指定起始日期和周期数
dates_periods = pd.date_range(start='2024-01-01', periods=7, freq='W-MON')
print("\n每周一 (freq='W-MON', periods=7):")
print(dates_periods)

# 常用频率说明:
# D=日, B=工作日, W=周, ME=月末, MS=月初, QE=季末, YE=年末, H=小时, T/min=分钟, S=秒

# ============================================================
# 2. pd.to_datetime() —— 将字符串转为日期时间
# ============================================================
print("\n" + "=" * 60)
print("2. pd.to_datetime() 字符串转日期时间")
print("=" * 60)

# 单个字符串转换
dt_str = pd.to_datetime('2024-03-15')
print(f"单个字符串转换: {dt_str}, 类型: {type(dt_str)}")

# 列表转换
dt_list = pd.to_datetime(['2024/01/01', '2024-02-15', 'Mar 20, 2024'])
print(f"列表转换:\n{dt_list}")

# 指定格式解析
dt_format = pd.to_datetime('15-03-2024', format='%d-%m-%Y')
print(f"指定格式解析: {dt_format}")

# 处理缺失值
dt_with_na = pd.to_datetime(['2024-01-01', None, '2024-01-03'])
print(f"含缺失值转换:\n{dt_with_na}")

# ============================================================
# 3. DatetimeIndex —— 日期时间索引
# ============================================================
print("\n" + "=" * 60)
print("3. DatetimeIndex 日期时间索引")
print("=" * 60)

dt_index = pd.DatetimeIndex(
    pd.date_range(start='2024-01-01', periods=5, freq='D')
)
ts = pd.Series([10, 20, 15, 25, 30], index=dt_index)
print("DatetimeIndex 作为索引的 Series:")
print(ts)

# DatetimeIndex 的属性
print(f"\n年份: {dt_index.year}")
print(f"月份: {dt_index.month}")
print(f"日期: {dt_index.day}")
print(f"星期几 (0=周一): {dt_index.dayofweek}")
print(f"季度: {dt_index.quarter}")
print(f"是否月初: {dt_index.is_month_start}")

# ============================================================
# 4. 设置时间索引
# ============================================================
print("\n" + "=" * 60)
print("4. 设置时间索引")
print("=" * 60)

df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'value': [100, 105, 98, 110, 108]
})
print("原始 DataFrame:")
print(df)
print(f"索引类型: {type(df.index)}")

# 方式1: set_index + to_datetime
df_indexed = df.copy()
df_indexed['date'] = pd.to_datetime(df_indexed['date'])
df_indexed = df_indexed.set_index('date')
print("\n方式1 - set_index:")
print(df_indexed)
print(f"索引类型: {type(df_indexed.index)}")

# 方式2: 直接在创建时指定 index
df_direct = pd.DataFrame(
    {'value': [100, 105, 98, 110, 108]},
    index=pd.date_range('2024-01-01', periods=5, freq='D')
)
print("\n方式2 - 创建时指定 index:")
print(df_direct)

# ============================================================
# 5. resample() —— 重采样
# ============================================================
print("\n" + "=" * 60)
print("5. resample() 重采样")
print("=" * 60)

# 生成每日数据
np.random.seed(42)
daily_dates = pd.date_range('2024-01-01', periods=90, freq='D')
daily_data = pd.Series(
    np.random.randn(90).cumsum() + 50,
    index=daily_dates
)
print("原始每日数据 (前5行):")
print(daily_data.head())

# 降采样: 日 -> 周，取均值
weekly_mean = daily_data.resample('W').mean()
print("\n降采样 日->周 (均值, 前3行):")
print(weekly_mean.head(3))

# 降采样: 日 -> 月，取多种聚合
monthly_stats = daily_data.resample('ME').agg(['mean', 'max', 'min'])
print("\n降采样 日->月 (多聚合):")
print(monthly_stats)

# 升采样: 日 -> 小时，前向填充
hourly_data = daily_data.iloc[:3].resample('h').ffill()
print("\n升采样 日->小时 (前向填充, 前6行):")
print(hourly_data.head(6))

# ============================================================
# 6. asfreq() —— 频率转换
# ============================================================
print("\n" + "=" * 60)
print("6. asfreq() 频率转换")
print("=" * 60)

# asfreq 与 resample 的区别:
# asfreq 仅做频率转换，不做聚合；resample 可做聚合运算

# 工作日频率 (跳过周末)
biz_data = daily_data.asfreq('B')
print(f"原始数据长度: {len(daily_data)}")
print(f"工作日频率长度: {len(biz_data)}")

# 填充方式
asfreq_ffill = daily_data.asfreq('12h', method='ffill')
print("\nasfreq 12小时频率 (前向填充, 前6行):")
print(asfreq_ffill.head(6))

# ============================================================
# 7. shift() —— 数据平移
# ============================================================
print("\n" + "=" * 60)
print("7. shift() 数据平移")
print("=" * 60)

ts_shift = pd.Series(
    [10, 15, 12, 18, 20],
    index=pd.date_range('2024-01-01', periods=5, freq='D')
)
print("原始数据:")
print(ts_shift)

# 向后平移1期 (数据向下移动)
print("\nshift(1) 向后平移1期:")
print(ts_shift.shift(1))

# 向前平移1期 (数据向上移动)
print("\nshift(-1) 向前平移1期:")
print(ts_shift.shift(-1))

# shift 的常见用途: 计算同比/环比
ts_lag1 = ts_shift.shift(1)
pct_change = (ts_shift - ts_lag1) / ts_lag1
print("\n环比增长率:")
print(pct_change)

# ============================================================
# 8. diff() —— 差分
# ============================================================
print("\n" + "=" * 60)
print("8. diff() 差分运算")
print("=" * 60)

print("原始数据:")
print(ts_shift)

# 一阶差分
print("\n一阶差分 diff(1):")
print(ts_shift.diff(1))

# 二阶差分
print("\n二阶差分 diff(2):")
print(ts_shift.diff(2))

# diff 与 shift 的关系: diff(1) = ts - ts.shift(1)
print("\n验证 diff(1) == ts - ts.shift(1):")
print(ts_shift.diff(1) == (ts_shift - ts_shift.shift(1)))

# ============================================================
# 9. 时间序列切片
# ============================================================
print("\n" + "=" * 60)
print("9. 时间序列切片")
print("=" * 60)

# 生成一年数据
np.random.seed(42)
year_data = pd.Series(
    np.random.randn(365).cumsum() + 100,
    index=pd.date_range('2024-01-01', periods=365, freq='D')
)

# 按字符串切片
print("2024年1月数据 (前5行):")
print(year_data['2024-01'].head())

# 按日期范围切片
print("\n2024-01-15 到 2024-01-20:")
print(year_data['2024-01-15':'2024-01-20'])

# 按年切片
print("\n2024年数据统计:")
print(year_data['2024'].describe())

# 按月份范围切片
print("\n第一季度数据统计:")
print(year_data['2024-01':'2024-03'].describe())

# 使用 truncate 截断
print("\ntruncate 保留 2024-06-01 之后的数据 (前3行):")
print(year_data.truncate(before='2024-06-01').head(3))

# ============================================================
# 10. 综合可视化
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 原始数据与重采样
daily_data.plot(ax=axes[0], title='每日数据 vs 周均值', alpha=0.5, label='每日')
daily_data.resample('W').mean().plot(ax=axes[0], label='周均值', linewidth=2)
axes[0].legend()

# shift 与 diff
ts_demo = pd.Series(
    np.random.randn(30).cumsum() + 50,
    index=pd.date_range('2024-01-01', periods=30, freq='D')
)
ts_demo.plot(ax=axes[1], label='原始', title='shift 与 diff')
ts_demo.shift(7).plot(ax=axes[1], label='shift(7)', alpha=0.7)
axes[1].legend()

ts_demo.diff().plot(ax=axes[2], label='一阶差分', title='一阶差分', color='green')
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/1_时间序列数据基础.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存。")
