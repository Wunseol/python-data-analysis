# 数据来源: 脚本内自建示例数据 (含故意设置的数据问题)
# 学习场景: Pandas API 速查 — 逐个演示数据清洗相关方法的基本用法
# 综合实战场景: 详见 03_数据清洗/ 目录，包含完整清洗流程、策略对比和综合案例

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、创建含问题的示例数据")
print("=" * 60)

df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十', '郑十一', '王五'],
    '年龄': [25, 30, np.nan, 35, -5, 28, 200, np.nan, 22, 30],
    '薪资': ['15000', '22000', '18000', 'N/A', '12000', '20000', '23000', '16000', 'unknown', '22000'],
    '入职日期': ['2019-03-15', '2017/08/20', '2020-01-10', '2016-11-05', '2022-06-01',
                '2018-09-12', '2015-04-20', '2021-12-30', '2023-02-14', '2017/08/20'],
    '部门': ['技术部', '市场部', '技术部', '财务部', '市场部', '技术部', '财务部', '市场部', '技术部', '市场部'],
    '绩效评级': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B'],
    '电话': ['138-0000-1234', '13911112222', '137-5555-6789', '13600001111',
             '135-2222-3333', '13444445555', '133-6666-7777', '13288889999',
             '131-0000-1111', '13911112222']
})

print("原始数据 (含问题):")
print(df.to_string())
print(f"\n数据形状: {df.shape}")

print("\n" + "=" * 60)
print("二、缺失值检测")
print("=" * 60)

# isnull / isna
print("\nisnull() 检测缺失值:")
print(df.isnull())

print("\n每列缺失值数量:")
print(df.isnull().sum())

print("\n缺失值比例:")
print((df.isnull().mean() * 100).round(2))

print("\n包含缺失值的行:")
print(df[df.isnull().any(axis=1)].to_string())

print("\nnotnull() 非缺失值:")
print(df.notnull().sum())

print("\n" + "=" * 60)
print("三、dropna() 删除缺失值")
print("=" * 60)

# 删除包含缺失值的行
df_drop_row = df.copy()
print(f"\ndropna() 删除含缺失值的行 (原{len(df_drop_row)}行):")
print(f"删除后: {len(df_drop_row.dropna())}行")

# 删除包含缺失值的列
print(f"\ndropna(axis=1) 删除含缺失值的列 (原{len(df.columns)}列):")
dropped_cols = df_drop_row.dropna(axis=1)
print(f"删除后: {len(dropped_cols.columns)}列, 保留列: {dropped_cols.columns.tolist()}")

# how='all' 仅删除全为缺失值的行
print(f"\ndropna(how='all') 仅删除全为缺失值的行:")
print(f"删除后: {len(df_drop_row.dropna(how='all'))}行")

# thresh 参数 - 保留至少有N个非缺失值的行
print(f"\ndropna(thresh=6) 保留至少6个非缺失值的行:")
print(f"删除后: {len(df_drop_row.dropna(thresh=6))}行")

# subset 参数 - 只检查特定列
print(f"\ndropna(subset=['年龄']) 只根据年龄列删除:")
result = df_drop_row.dropna(subset=['年龄'])
print(result[['姓名', '年龄']].to_string())

print("\n" + "=" * 60)
print("四、fillna() 填充缺失值")
print("=" * 60)

df_fill = df.copy()

# 用固定值填充
df_fill['年龄_固定填充'] = df_fill['年龄'].fillna(0)
print("\n用0填充年龄缺失值:")
print(df_fill[['姓名', '年龄', '年龄_固定填充']].to_string())

# 用均值填充
age_mean = df_fill['年龄'].mean()
df_fill['年龄_均值填充'] = df_fill['年龄'].fillna(age_mean)
print(f"\n用均值({age_mean:.1f})填充年龄缺失值:")
print(df_fill[['姓名', '年龄', '年龄_均值填充']].to_string())

# 用中位数填充
age_median = df_fill['年龄'].median()
df_fill['年龄_中位数填充'] = df_fill['年龄'].fillna(age_median)
print(f"\n用中位数({age_median:.1f})填充年龄缺失值:")
print(df_fill[['姓名', '年龄', '年龄_中位数填充']].to_string())

# 用众数填充
mode_val = df_fill['部门'].mode()[0]
print(f"\n部门列的众数: {mode_val}")

# ffill 前向填充
df_fill['年龄_ffill'] = df_fill['年龄'].ffill()
print("\n前向填充 (ffill):")
print(df_fill[['姓名', '年龄', '年龄_ffill']].to_string())

# bfill 后向填充
df_fill['年龄_bfill'] = df_fill['年龄'].bfill()
print("\n后向填充 (bfill):")
print(df_fill[['姓名', '年龄', '年龄_bfill']].to_string())

# 用字典对不同列填充不同值
df_fill_dict = df.fillna({'年龄': df['年龄'].median(), '薪资': '0'})
print("\n字典方式填充不同列:")
print(df_fill_dict[['姓名', '年龄', '薪资']].to_string())

print("\n" + "=" * 60)
print("五、重复值处理")
print("=" * 60)

# 检测重复行
print("\nduplicated() 检测重复行:")
print(df.duplicated())

print(f"\n重复行数量: {df.duplicated().sum()}")
print("重复行内容:")
print(df[df.duplicated()].to_string())

# drop_duplicates 删除重复行
df_dedup = df.drop_duplicates()
print(f"\ndrop_duplicates() 删除重复行 (原{len(df)}行 → {len(df_dedup)}行)")

# 基于特定列去重
df_dedup_subset = df.drop_duplicates(subset=['姓名'])
print(f"\n基于姓名列去重: {len(df_dedup_subset)}行")

# keep 参数
df_dedup_last = df.drop_duplicates(subset=['姓名'], keep='last')
print(f"\nkeep='last' 保留最后出现: {len(df_dedup_last)}行")

df_dedup_none = df.drop_duplicates(subset=['姓名'], keep=False)
print(f"\nkeep=False 删除所有重复: {len(df_dedup_none)}行")

print("\n" + "=" * 60)
print("六、数据类型转换 astype")
print("=" * 60)

df_clean = df.drop_duplicates().copy()

# 查看当前类型
print("\n当前数据类型:")
print(df_clean.dtypes)

# 转换类型
df_clean['年龄'] = df_clean['年龄'].fillna(df_clean['年龄'].median()).astype(int)
print(f"\n年龄列转为 int:")
print(df_clean['年龄'].dtype)

# category 类型
df_clean['部门'] = df_clean['部门'].astype('category')
print(f"\n部门列转为 category:")
print(df_clean['部门'].dtype)
print(f"分类类别: {df_clean['部门'].cat.categories.tolist()}")

# 绩效评级转 category 并指定顺序
df_clean['绩效评级'] = pd.Categorical(df_clean['绩效评级'], categories=['C', 'B', 'A'], ordered=True)
print(f"\n绩效评级有序 category:")
print(df_clean['绩效评级'].dtype)
print(f"排序后:")
print(df_clean.sort_values('绩效评级')[['姓名', '绩效评级']].to_string())

print("\n" + "=" * 60)
print("七、to_numeric 类型转换")
print("=" * 60)

# 薪资列包含 'N/A' 和 'unknown'，无法直接转数值
print("\n原始薪资列:")
print(df_clean['薪资'].tolist())

# errors='coerce' 将无法转换的设为 NaN
df_clean['薪资'] = pd.to_numeric(df_clean['薪资'], errors='coerce')
print("\nto_numeric(errors='coerce') 后:")
print(df_clean['薪资'].tolist())
print(f"缺失值数量: {df_clean['薪资'].isna().sum()}")

# 填充转换后的缺失值
df_clean['薪资'] = df_clean['薪资'].fillna(df_clean['薪资'].mean()).astype(int)
print(f"\n填充后薪资列:")
print(df_clean['薪资'].tolist())

print("\n" + "=" * 60)
print("八、to_datetime 日期转换")
print("=" * 60)

# 原始日期格式不统一
print("\n原始入职日期列:")
print(df_clean['入职日期'].tolist())

df_clean['入职日期'] = pd.to_datetime(df_clean['入职日期'], format='mixed')
print("\nto_datetime 转换后:")
print(df_clean['入职日期'].tolist())
print(f"类型: {df_clean['入职日期'].dtype}")

# 提取日期信息
df_clean['入职年份'] = df_clean['入职日期'].dt.year
df_clean['入职月份'] = df_clean['入职日期'].dt.month
print("\n提取年份和月份:")
print(df_clean[['姓名', '入职日期', '入职年份', '入职月份']].to_string())

print("\n" + "=" * 60)
print("九、异常值处理")
print("=" * 60)

# 年龄列有异常值 (-5, 200)
print("\n年龄列统计:")
print(df_clean['年龄'].describe())

# 使用条件过滤异常值
age_mask = (df_clean['年龄'] >= 18) & (df_clean['年龄'] <= 65)
df_clean = df_clean[age_mask]
print(f"\n过滤年龄异常值后 (18-65): {len(df_clean)}行")
print(df_clean[['姓名', '年龄']].to_string())

print("\n" + "=" * 60)
print("十、清洗后数据概览")
print("=" * 60)

print(f"\n清洗后数据形状: {df_clean.shape}")
print(f"\n数据类型:\n{df_clean.dtypes}")
print(f"\n缺失值:\n{df_clean.isnull().sum()}")
print(f"\n清洗后数据:")
print(df_clean.to_string())
