# 数据来源: 脚本内自建示例数据
# 学习场景: Pandas API 速查 — 逐个演示时间序列处理方法的基本用法
# 综合实战场景: 详见 04_分析建模/06_时序分析与预测/ 目录，包含时序分解、ARIMA、Prophet 等建模方法

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、pd.to_datetime() 日期转换")
print("=" * 60)

# 字符串转日期
date_str = pd.Series(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05'])
dates = pd.to_datetime(date_str)
print("\n字符串转日期:")
print(dates)

# 不同格式
dates_mixed = pd.to_datetime(['2024/01/15', 'Jan 15, 2024', '15-01-2024'], format='mixed')
print("\n不同格式转日期:")
print(dates_mixed)

# errors 参数
dates_coerce = pd.to_datetime(['2024-01-15', 'invalid', '2024-03-10'], errors='coerce')
print("\nerrors='coerce' 无效日期变 NaT:")
print(dates_coerce)

# 从 DataFrame 列转换
df = pd.DataFrame({
    '日期字符串': ['2024-01-01', '2024-02-14', '2024-03-08', '2024-04-22', '2024-05-01'],
    '值': [100, 120, 115, 130, 125]
})
df['日期'] = pd.to_datetime(df['日期字符串'])
print(f"\nDataFrame 日期列转换, 类型: {df['日期'].dtype}")
print(df)

print("\n" + "=" * 60)
print("二、dt 访问器")
print("=" * 60)

dates = pd.date_range('2024-01-01', periods=10, freq='D')
s = pd.Series(dates)

print(f"\n年 (dt.year): {s.dt.year.tolist()}")
print(f"月 (dt.month): {s.dt.month.tolist()}")
print(f"日 (dt.day): {s.dt.day.tolist()}")
print(f"星期几 (dt.dayofweek, 0=周一): {s.dt.dayofweek.tolist()}")
print(f"星期名称 (dt.day_name): {s.dt.day_name().tolist()}")
print(f"季度 (dt.quarter): {s.dt.quarter.tolist()}")
print(f"一年中第几天 (dt.dayofyear): {s.dt.dayofyear.tolist()}")
print(f"是否月初 (dt.is_month_start): {s.dt.is_month_start.tolist()}")
print(f"是否月末 (dt.is_month_end): {s.dt.is_month_end.tolist()}")

# 时间差
s_time = pd.Series(pd.to_datetime(['2024-01-15 08:30:00', '2024-01-15 14:45:30', '2024-01-15 23:59:59']))
print(f"\n小时 (dt.hour): {s_time.dt.hour.tolist()}")
print(f"分钟 (dt.minute): {s_time.dt.minute.tolist()}")
print(f"秒 (dt.second): {s_time.dt.second.tolist()}")

print("\n" + "=" * 60)
print("三、pd.date_range() 生成日期范围")
print("=" * 60)

# 指定起止和频率
dr1 = pd.date_range('2024-01-01', '2024-01-10', freq='D')
print(f"\n日频率 (D): {dr1.tolist()[:5]}...")

# 指定起止和数量
dr2 = pd.date_range('2024-01-01', periods=5, freq='D')
print(f"\n5天: {dr2.tolist()}")

# 工作日频率
dr_bday = pd.date_range('2024-01-01', periods=5, freq='B')
print(f"\n工作日 (B): {dr_bday.tolist()}")

# 月初频率
dr_ms = pd.date_range('2024-01-01', periods=5, freq='MS')
print(f"\n月初 (MS): {dr_ms.tolist()}")

# 月末频率
dr_me = pd.date_range('2024-01-01', periods=5, freq='ME')
print(f"\n月末 (ME): {dr_me.tolist()}")

# 小时频率
dr_h = pd.date_range('2024-01-01', periods=5, freq='h')
print(f"\n小时 (h): {dr_h.tolist()}")

# 季度频率
dr_q = pd.date_range('2024-01-01', periods=4, freq='QE')
print(f"\n季末 (QE): {dr_q.tolist()}")

print("\n" + "=" * 60)
print("四、创建时间序列 DataFrame")
print("=" * 60)

np.random.seed(42)
date_idx = pd.date_range('2024-01-01', periods=100, freq='D')
df_ts = pd.DataFrame({
    '日期': date_idx,
    '销售额': np.random.normal(5000, 1000, 100).cumsum(),
    '访客数': np.random.randint(100, 500, 100),
    '转化率': np.random.uniform(0.02, 0.08, 100)
})
df_ts = df_ts.set_index('日期')
print(f"\n时间序列数据 (前10行):")
print(df_ts.head(10).round(2))

print(f"\n索引类型: {type(df_ts.index)}")
print(f"索引频率推断: {pd.infer_freq(df_ts.index)}")

print("\n" + "=" * 60)
print("五、resample() 重采样")
print("=" * 60)

# 日 → 周汇总
weekly = df_ts['销售额'].resample('W').mean()
print("\n日数据重采样为周均值 (前5周):")
print(weekly.head().round(2))

# 日 → 月汇总
monthly = df_ts['销售额'].resample('ME').agg(['mean', 'sum', 'count'])
monthly.columns = ['月均', '月合计', '天数']
print("\n日数据重采样为月汇总:")
print(monthly.round(2))

# 日 → 季度汇总
quarterly = df_ts['销售额'].resample('QE').mean()
print("\n日数据重采样为季度均值:")
print(quarterly.round(2))

# 升采样 (填充)
df_sparse = pd.DataFrame({
    '值': [10, 20, 30]
}, index=pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10']))

upsampled = df_sparse.resample('D').ffill()
print("\n升采样 (5天→1天, ffill填充):")
print(upsampled.head(10))

print("\n" + "=" * 60)
print("六、asfreq() 频率转换")
print("=" * 60)

# asfreq 只改变频率，不做聚合
df_af = pd.DataFrame({
    '值': [1, 2, 3, 4, 5]
}, index=pd.date_range('2024-01-01', periods=5, freq='D'))

print("\n原始日频率数据:")
print(df_af)

print("\nasfreq('h') 转为小时频率 (产生NaN):")
print(df_af.asfreq('h').head(10))

print("\nasfreq('h', fill_value=0) 填充0:")
print(df_af.asfreq('h', fill_value=0).head(10))

print("\n" + "=" * 60)
print("七、shift() 与 diff()")
print("=" * 60)

df_shift = df_ts[['销售额']].head(10).copy()

# shift 向下移动
df_shift['销售额_前1天'] = df_shift['销售额'].shift(1)
df_shift['销售额_前3天'] = df_shift['销售额'].shift(3)

# shift 向上移动
df_shift['销售额_后1天'] = df_shift['销售额'].shift(-1)

print("\nshift 移动数据:")
print(df_shift.round(2).to_string())

# diff 差分
df_diff = df_ts[['销售额']].head(10).copy()
df_diff['日变化量'] = df_diff['销售额'].diff()
df_diff['日变化率'] = df_diff['销售额'].pct_change()

print("\ndiff 差分:")
print(df_diff.round(4).to_string())

# 多步差分
df_diff['3日差分'] = df_diff['销售额'].diff(3)
print("\n3日差分:")
print(df_diff[['销售额', '3日差分']].round(2).to_string())

print("\n" + "=" * 60)
print("八、pct_change() 百分比变化")
print("=" * 60)

df_pct = df_ts[['销售额']].head(10).copy()
df_pct['日涨跌幅'] = df_pct['销售额'].pct_change()
df_pct['5日涨跌幅'] = df_pct['销售额'].pct_change(periods=5)

print("\npct_change 百分比变化:")
print(df_pct.round(4).to_string())

print("\n" + "=" * 60)
print("九、rolling() 滚动窗口")
print("=" * 60)

df_roll = df_ts[['销售额']].head(20).copy()

# 滚动均值
df_roll['7日均线'] = df_roll['销售额'].rolling(window=7).mean()
df_roll['14日均线'] = df_roll['销售额'].rolling(window=14).mean()

print("\n滚动窗口均值:")
print(df_roll.round(2).to_string())

# 滚动标准差
df_roll['7日标准差'] = df_roll['销售额'].rolling(window=7).std()

# 滚动最大最小值
df_roll['7日最大值'] = df_roll['销售额'].rolling(window=7).max()
df_roll['7日最小值'] = df_roll['销售额'].rolling(window=7).min()

print("\n滚动窗口统计:")
print(df_roll[['销售额', '7日均线', '7日标准差', '7日最大值', '7日最小值']].round(2).to_string())

# rolling 自定义函数
df_roll['7日均值±标准差'] = df_roll['销售额'].rolling(window=7).apply(
    lambda x: f'{x.mean():.0f}±{x.std():.0f}', raw=True)

# min_periods 参数
df_roll['7日均线_min3'] = df_roll['销售额'].rolling(window=7, min_periods=3).mean()
print("\nmin_periods=3 (至少3个值即计算):")
print(df_roll[['销售额', '7日均线', '7日均线_min3']].round(2).to_string())

# expanding 累计窗口
df_roll['累计均值'] = df_roll['销售额'].expanding().mean()
print("\nexpanding 累计均值:")
print(df_roll[['销售额', '累计均值']].round(2).to_string())

print("\n" + "=" * 60)
print("十、DatetimeIndex 操作")
print("=" * 60)

# 按日期筛选
print("\n2024年2月的数据:")
print(df_ts.loc['2024-02'].head().round(2))

print("\n2024年1月15日到1月20日的数据:")
print(df_ts.loc['2024-01-15':'2024-01-20'].round(2))

# 按月份切片
print("\n2024年1-2月的数据 (前5行):")
print(df_ts.loc['2024-01':'2024-02'].head().round(2))

# 时间差计算
date1 = pd.Timestamp('2024-06-15')
date2 = pd.Timestamp('2024-01-01')
delta = date1 - date2
print(f"\n时间差: {delta}")
print(f"天数: {delta.days}")

# 日期偏移
base_date = pd.Timestamp('2024-01-01')
print(f"\n基准日期: {base_date}")
print(f"加10个工作日: {base_date + pd.offsets.BDay(10)}")
print(f"加2个月: {base_date + pd.offsets.MonthEnd(2)}")
print(f"加1个季度: {base_date + pd.offsets.QuarterEnd(1)}")
