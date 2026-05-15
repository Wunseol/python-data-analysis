# 数据来源: 脚本内自建示例数据
# 本脚本演示 Pandas 数据合并 (merge) 与拼接 (concat) 操作

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

df_employees = pd.DataFrame({
    '员工ID': ['E001', 'E002', 'E003', 'E004', 'E005'],
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '部门ID': ['D01', 'D02', 'D01', 'D03', 'D02']
})

df_departments = pd.DataFrame({
    '部门ID': ['D01', 'D02', 'D03', 'D04'],
    '部门名称': ['技术部', '市场部', '财务部', '人事部'],
    '负责人': ['刘总', '陈总', '杨总', '黄总']
})

df_salaries = pd.DataFrame({
    '员工ID': ['E001', 'E002', 'E003', 'E004', 'E005'],
    '基本工资': [10000, 12000, 11000, 15000, 9000],
    '奖金': [3000, 5000, 4000, 6000, 2000]
})

print("员工表:")
print(df_employees)
print("\n部门表:")
print(df_departments)
print("\n薪资表:")
print(df_salaries)

print("\n" + "=" * 60)
print("二、pd.merge() 基本合并")
print("=" * 60)

# inner join (默认)
result_inner = pd.merge(df_employees, df_departments, on='部门ID', how='inner')
print("\ninner join (默认):")
print(result_inner)

# left join
result_left = pd.merge(df_employees, df_departments, on='部门ID', how='left')
print("\nleft join:")
print(result_left)

# right join
result_right = pd.merge(df_employees, df_departments, on='部门ID', how='right')
print("\nright join:")
print(result_right)

# outer join
result_outer = pd.merge(df_employees, df_departments, on='部门ID', how='outer')
print("\nouter join:")
print(result_outer)

print("\n" + "=" * 60)
print("三、merge 参数详解")
print("=" * 60)

# on 参数 - 单列连接
print("\non='员工ID' 单列连接:")
result_on = pd.merge(df_employees, df_salaries, on='员工ID')
print(result_on)

# 不同列名连接
df_employees2 = df_employees.copy()
df_employees2 = df_employees2.rename(columns={'员工ID': 'emp_id'})
df_salaries2 = df_salaries.copy()
df_salaries2 = df_salaries2.rename(columns={'员工ID': 'emp_id'})

result_diff_col = pd.merge(df_employees2, df_salaries2, left_on='emp_id', right_on='emp_id')
print("\nleft_on / right_on 不同列名连接:")
print(result_diff_col)

# indicator 参数 - 显示合并来源
result_indicator = pd.merge(df_employees, df_departments, on='部门ID', how='outer', indicator=True)
print("\nindicator=True 显示合并来源:")
print(result_indicator)

# suffixes 参数 - 重名列后缀
df_eval1 = pd.DataFrame({
    '员工ID': ['E001', 'E002', 'E003'],
    '评分': [85, 90, 78],
    '评语': ['优秀', '卓越', '良好']
})
df_eval2 = pd.DataFrame({
    '员工ID': ['E001', 'E002', 'E003'],
    '评分': [88, 85, 82],
    '评语': ['出色', '优秀', '不错']
})

result_suffix = pd.merge(df_eval1, df_eval2, on='员工ID', suffixes=('_上半年', '_下半年'))
print("\nsuffixes 处理重名列:")
print(result_suffix)

print("\n" + "=" * 60)
print("四、多列连接")
print("=" * 60)

df_sales1 = pd.DataFrame({
    '地区': ['华东', '华东', '华北', '华北'],
    '产品': ['A', 'B', 'A', 'B'],
    'Q1销量': [100, 200, 150, 180]
})
df_sales2 = pd.DataFrame({
    '地区': ['华东', '华东', '华北', '华北'],
    '产品': ['A', 'B', 'A', 'B'],
    'Q2销量': [120, 210, 160, 190]
})

result_multi = pd.merge(df_sales1, df_sales2, on=['地区', '产品'])
print("\n多列连接:")
print(result_multi)

print("\n" + "=" * 60)
print("五、pd.concat() 拼接")
print("=" * 60)

df_q1 = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '销售额': [5000, 6000, 4500]
})
df_q2 = pd.DataFrame({
    '姓名': ['赵六', '钱七', '孙八'],
    '销售额': [7000, 5500, 4800]
})

# 纵向拼接 (axis=0, 默认)
result_concat_row = pd.concat([df_q1, df_q2])
print("\n纵向拼接 (默认):")
print(result_concat_row)

# ignore_index=True 重置索引
result_concat_reset = pd.concat([df_q1, df_q2], ignore_index=True)
print("\nignore_index=True:")
print(result_concat_reset)

# keys 参数 - 多层索引标识来源
result_concat_keys = pd.concat([df_q1, df_q2], keys=['Q1', 'Q2'])
print("\nkeys 标识来源:")
print(result_concat_keys)

# 横向拼接 (axis=1)
df_info = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '部门': ['技术部', '市场部', '财务部']
})
df_score = pd.DataFrame({
    '绩效': [85, 92, 78],
    '考勤': [100, 95, 98]
})

result_concat_col = pd.concat([df_info, df_score], axis=1)
print("\n横向拼接 (axis=1):")
print(result_concat_col)

print("\n" + "=" * 60)
print("六、concat 处理列不一致")
print("=" * 60)

df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_b = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})

# 默认: 缺失列填充 NaN
result_diff_cols = pd.concat([df_a, df_b], ignore_index=True)
print("\n列不一致时拼接 (缺失填充NaN):")
print(result_diff_cols)

# join='inner' 只保留共有列
result_inner_cols = pd.concat([df_a, df_b], ignore_index=True, join='inner')
print("\njoin='inner' 只保留共有列:")
print(result_inner_cols)

print("\n" + "=" * 60)
print("七、merge_ordered 与 merge_asof 概念")
print("=" * 60)

# merge_ordered: 有序合并，支持 fill_method
df_price = pd.DataFrame({
    '日期': pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05']),
    '价格': [100, 105, 110]
})
df_volume = pd.DataFrame({
    '日期': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-04']),
    '成交量': [1000, 1200, 800]
})

result_ordered = pd.merge_ordered(df_price, df_volume, on='日期', fill_method='ffill')
print("\nmerge_ordered (fill_method='ffill'):")
print(result_ordered)

# merge_asof: 近似匹配合并
df_trades = pd.DataFrame({
    '时间': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:30:05', '2024-01-01 09:30:12']),
    '价格': [100.0, 100.5, 101.0]
})
df_quotes = pd.DataFrame({
    '时间': pd.to_datetime(['2024-01-01 09:30:00', '2024-01-01 09:30:03', '2024-01-01 09:30:10']),
    '报价': [99.5, 100.2, 100.8]
})

result_asof = pd.merge_asof(df_trades, df_quotes, on='时间', direction='nearest')
print("\nmerge_asof (direction='nearest'):")
print(result_asof)

print("\n" + "=" * 60)
print("八、join() 方法")
print("=" * 60)

df_left = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df_right = pd.DataFrame({'B': [4, 5]}, index=['a', 'b'])

result_join = df_left.join(df_right)
print("\njoin() 默认左连接:")
print(result_join)

result_join_inner = df_left.join(df_right, how='inner')
print("\njoin(how='inner'):")
print(result_join_inner)

result_join_outer = df_left.join(df_right, how='outer')
print("\njoin(how='outer'):")
print(result_join_outer)
