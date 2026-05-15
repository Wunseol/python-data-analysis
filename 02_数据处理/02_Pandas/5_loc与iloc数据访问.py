# 数据来源: 自建示例数据
# 本脚本演示 Pandas loc 与 iloc 数据访问方法

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'],
    '部门': ['技术部', '市场部', '技术部', '财务部', '市场部', '技术部', '财务部', '市场部'],
    '薪资': [15000, 22000, 18000, 25000, 12000, 20000, 23000, 16000],
    '绩效评分': [85, 92, 78, 88, 65, 90, 95, 72],
    '入职年份': [2019, 2017, 2020, 2016, 2022, 2018, 2015, 2021]
}, index=['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008'])

print(df)

print("\n" + "=" * 60)
print("二、loc 基于标签的访问")
print("=" * 60)

# 选取单行
print("\nloc['E001'] 选取单行:")
print(df.loc['E001'])

# 选取多行
print("\nloc[['E001', 'E003', 'E005']] 选取多行:")
print(df.loc[['E001', 'E003', 'E005']])

# 行列同时选取
print("\nloc['E001', '姓名'] 行列同时选取:")
print(df.loc['E001', '姓名'])

print("\nloc[['E001', 'E002'], ['姓名', '薪资']]:")
print(df.loc[['E001', 'E002'], ['姓名', '薪资']])

# 切片（标签切片包含末端）
print("\nloc['E001':'E003', '姓名':'薪资'] 标签切片:")
print(df.loc['E001':'E003', '姓名':'薪资'])

# 选取所有行的部分列
print("\nloc[:, ['姓名', '薪资']] 所有行+部分列:")
print(df.loc[:, ['姓名', '薪资']])

# 使用条件筛选
print("\nloc[df['薪资'] > 18000] 条件筛选:")
print(df.loc[df['薪资'] > 18000])

# 条件筛选 + 指定列
print("\nloc[df['薪资'] > 18000, ['姓名', '薪资', '部门']]:")
print(df.loc[df['薪资'] > 18000, ['姓名', '薪资', '部门']])

# loc 赋值
df_loc = df.copy()
df_loc.loc['E001', '薪资'] = 16000
print("\nloc 赋值后 E001 薪资:")
print(df_loc.loc['E001', '薪资'])

# loc 新增行
df_loc.loc['E009'] = ['郑十一', '技术部', 17000, 80, 2023]
print("\nloc 新增行 E009:")
print(df_loc.loc['E009'])

print("\n" + "=" * 60)
print("三、iloc 基于位置的访问")
print("=" * 60)

# 选取单行
print("\niloc[0] 选取第1行:")
print(df.iloc[0])

# 选取多行
print("\niloc[[0, 2, 4]] 选取第1、3、5行:")
print(df.iloc[[0, 2, 4]])

# 切片（位置切片不包含末端）
print("\niloc[0:3] 前3行 (不包含索引3):")
print(df.iloc[0:3])

# 行列同时选取
print("\niloc[0, 1] 第1行第2列:")
print(df.iloc[0, 1])

print("\niloc[0:3, 0:2] 前3行前2列:")
print(df.iloc[0:3, 0:2])

print("\niloc[[0, 2], [1, 3]] 指定行列位置:")
print(df.iloc[[0, 2], [1, 3]])

# 选取所有行的部分列
print("\niloc[:, [0, 2]] 所有行+第1、3列:")
print(df.iloc[:, [0, 2]])

# 负数索引
print("\niloc[-1] 最后一行:")
print(df.iloc[-1])

print("\niloc[-3:] 最后3行:")
print(df.iloc[-3:])

# iloc 赋值
df_iloc = df.copy()
df_iloc.iloc[0, 2] = 16000
print(f"\niloc 赋值后第1行薪资: {df_iloc.iloc[0, 2]}")

print("\n" + "=" * 60)
print("四、布尔索引")
print("=" * 60)

# 单条件
print("\n薪资 > 18000:")
print(df[df['薪资'] > 18000])

# 多条件 AND
print("\n薪资 > 18000 且 部门为技术部:")
print(df[(df['薪资'] > 18000) & (df['部门'] == '技术部')])

# 多条件 OR
print("\n部门为技术部 或 部门为财务部:")
print(df[(df['部门'] == '技术部') | (df['部门'] == '财务部')])

# 取反
print("\n部门不为市场部:")
print(df[~(df['部门'] == '市场部')])

print("\n" + "=" * 60)
print("五、条件选择与 query()")
print("=" * 60)

# query 基本用法
print("\nquery('薪资 > 18000'):")
print(df.query('薪资 > 18000'))

# query 多条件
print("\nquery('薪资 > 18000 & 部门 == \"技术部\"'):")
print(df.query('薪资 > 18000 & 部门 == "技术部"'))

# query 使用变量
threshold = 18000
print(f"\nquery('薪资 > @threshold') (threshold={threshold}):")
print(df.query('薪资 > @threshold'))

# query 使用 in
print("\nquery('部门 in [\"技术部\", \"财务部\"]'):")
print(df.query('部门 in ["技术部", "财务部"]'))

print("\n" + "=" * 60)
print("六、isin() 筛选")
print("=" * 60)

# 单列 isin
print("\n部门 isin(['技术部', '财务部']):")
print(df[df['部门'].isin(['技术部', '财务部'])])

# 多列 isin
print("\n多列 isin 筛选:")
mask = df.isin({'部门': ['技术部'], '绩效评分': [85, 90, 95]})
print(df[mask.any(axis=1)])

# ~isin 取反
print("\n部门不在 ['市场部'] 中:")
print(df[~df['部门'].isin(['市场部'])])

print("\n" + "=" * 60)
print("七、多条件组合筛选")
print("=" * 60)

# 复杂条件
condition = (
    (df['薪资'] >= 15000) &
    (df['薪资'] <= 22000) &
    (df['绩效评分'] >= 80) &
    (df['入职年份'] <= 2020)
)
print("\n薪资 15000-22000, 绩效>=80, 2020年前入职:")
print(df[condition])

# 使用 between
print("\n薪资 between 15000 和 22000:")
print(df[df['薪资'].between(15000, 22000)])

# 使用 where (不满足条件的设为 NaN)
print("\nwhere(薪资 > 18000) 不满足条件设为 NaN:")
print(df['薪资'].where(df['薪资'] > 18000))

# 使用 mask (满足条件的设为 NaN)
print("\nmask(薪资 > 18000) 满足条件设为 NaN:")
print(df['薪资'].mask(df['薪资'] > 18000))
