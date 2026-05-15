# 数据来源: 脚本内自建示例数据
# 本脚本演示 Pandas 透视表 (pivot_table) 与交叉表 (crosstab)

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

np.random.seed(42)
df = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=40, freq='D'),
    '地区': np.random.choice(['华东', '华南', '华北', '西南'], 40),
    '产品类别': np.random.choice(['电子产品', '服装', '食品', '家居'], 40),
    '销售额': np.random.randint(1000, 20000, 40),
    '数量': np.random.randint(1, 50, 40),
    '销售员': np.random.choice(['张三', '李四', '王五', '赵六'], 40)
})

print(df.head(10).to_string())

print("\n" + "=" * 60)
print("二、pd.pivot_table() 基本用法")
print("=" * 60)

# 最简单的透视表
pt_simple = pd.pivot_table(df, values='销售额', index='地区', aggfunc='mean')
print("\n简单透视表 (按地区求平均销售额):")
print(pt_simple.round(2))

# 指定多个 index
pt_multi_index = pd.pivot_table(df, values='销售额', index=['地区', '产品类别'], aggfunc='mean')
print("\n多级索引透视表:")
print(pt_multi_index.round(2))

# 指定 columns
pt_columns = pd.pivot_table(df, values='销售额', index='地区', columns='产品类别', aggfunc='mean')
print("\n指定 columns 透视表:")
print(pt_columns.round(2))

print("\n" + "=" * 60)
print("三、pivot_table 参数详解")
print("=" * 60)

# 多个 values
pt_multi_values = pd.pivot_table(df, values=['销售额', '数量'], index='地区', aggfunc='mean')
print("\n多个 values:")
print(pt_multi_values.round(2))

# 多个 aggfunc
pt_multi_agg = pd.pivot_table(df, values='销售额', index='地区', columns='产品类别',
                               aggfunc=['mean', 'sum', 'count'])
print("\n多个 aggfunc:")
print(pt_multi_agg.round(2))

# 使用字典指定不同列的不同聚合函数
pt_dict_agg = pd.pivot_table(df, values=['销售额', '数量'], index='地区',
                              aggfunc={'销售额': ['mean', 'sum'], '数量': 'sum'})
print("\n字典指定不同聚合函数:")
print(pt_dict_agg.round(2))

# margins 参数 - 添加汇总行/列
pt_margins = pd.pivot_table(df, values='销售额', index='地区', columns='产品类别',
                             aggfunc='sum', margins=True, margins_name='合计')
print("\nmargins=True 添加汇总:")
print(pt_margins.round(2))

# fill_value 填充缺失值
pt_fill = pd.pivot_table(df, values='销售额', index='地区', columns='产品类别',
                          aggfunc='mean', fill_value=0)
print("\nfill_value=0 填充缺失值:")
print(pt_fill.round(2))

print("\n" + "=" * 60)
print("四、pivot_table 高级用法")
print("=" * 60)

# 自定义聚合函数
pt_custom = pd.pivot_table(df, values='销售额', index='地区',
                            aggfunc=lambda x: f'{x.mean():.0f} ± {x.std():.0f}')
print("\n自定义聚合函数 (均值±标准差):")
print(pt_custom)

# 多级 index + columns
pt_hierarchical = pd.pivot_table(df, values='销售额',
                                  index=['地区', '销售员'],
                                  columns='产品类别',
                                  aggfunc='sum',
                                  fill_value=0)
print("\n多级 index + columns:")
print(pt_hierarchical.to_string())

# 从透视表中提取数据
print(f"\n提取华东地区数据:")
print(pt_columns.loc['华东'].round(2) if '华东' in pt_columns.index else "华东不存在")

print("\n" + "=" * 60)
print("五、pd.crosstab() 交叉表")
print("=" * 60)

# 基本交叉表 - 频数统计
ct_basic = pd.crosstab(df['地区'], df['产品类别'])
print("\n基本交叉表 (频数统计):")
print(ct_basic)

# values 和 aggfunc
ct_values = pd.crosstab(df['地区'], df['产品类别'], values=df['销售额'], aggfunc='mean')
print("\n交叉表带 values 和 aggfunc:")
print(ct_values.round(2))

# normalize 参数 - 归一化
ct_norm_all = pd.crosstab(df['地区'], df['产品类别'], normalize='all')
print("\nnormalize='all' (占总体比例):")
print((ct_norm_all * 100).round(2))

ct_norm_index = pd.crosstab(df['地区'], df['产品类别'], normalize='index')
print("\nnormalize='index' (按行归一化):")
print((ct_norm_index * 100).round(2))

ct_norm_col = pd.crosstab(df['地区'], df['产品类别'], normalize='columns')
print("\nnormalize='columns' (按列归一化):")
print((ct_norm_col * 100).round(2))

# fill_value 填充缺失值
ct_fill = pd.crosstab(df['地区'], df['产品类别'], values=df['销售额'],
                       aggfunc='sum', fill_value=0)
print("\nfill_value=0:")
print(ct_fill)

# margins 参数
ct_margins = pd.crosstab(df['地区'], df['产品类别'], margins=True, margins_name='合计')
print("\nmargins=True:")
print(ct_margins)

print("\n" + "=" * 60)
print("六、交叉表多级维度")
print("=" * 60)

# 行多级
ct_multi_row = pd.crosstab([df['地区'], df['销售员']], df['产品类别'])
print("\n行多级交叉表:")
print(ct_multi_row.to_string())

# 列多级
ct_multi_col = pd.crosstab(df['地区'], [df['产品类别'], df['销售员']])
print("\n列多级交叉表:")
print(ct_multi_col.to_string())

print("\n" + "=" * 60)
print("七、透视表与交叉表对比")
print("=" * 60)

print("""
对比总结:
┌──────────────┬─────────────────────┬──────────────────────┐
│     特性      │    pivot_table      │      crosstab        │
├──────────────┼─────────────────────┼──────────────────────┤
│ 默认聚合函数  │    mean             │    count             │
│ 数据来源      │    DataFrame        │    Series/数组       │
│ 频数统计      │    需指定aggfunc    │    默认行为          │
│ 归一化        │    不支持           │    normalize参数     │
│ 适用场景      │    数值汇总分析     │    分类频数统计      │
└──────────────┴─────────────────────┴──────────────────────┘
""")

# 同一需求用两种方式实现
print("用 pivot_table 实现频数统计:")
print(pd.pivot_table(df, values='销售额', index='地区', columns='产品类别', aggfunc='count', fill_value=0))

print("\n用 crosstab 实现频数统计:")
print(pd.crosstab(df['地区'], df['产品类别']))
