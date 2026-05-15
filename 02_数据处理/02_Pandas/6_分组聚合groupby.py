# 数据来源: 脚本内自建示例数据
# 本脚本演示 Pandas GroupBy 分组聚合操作

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

np.random.seed(42)
df = pd.DataFrame({
    '部门': np.random.choice(['技术部', '市场部', '财务部', '人事部'], 20),
    '职级': np.random.choice(['初级', '中级', '高级'], 20),
    '姓名': [f'员工{i:02d}' for i in range(1, 21)],
    '薪资': np.random.randint(8000, 30000, 20),
    '绩效评分': np.random.randint(60, 100, 20),
    '工龄(年)': np.random.randint(1, 15, 20)
})

print(df.to_string())

print("\n" + "=" * 60)
print("二、基本 groupby 操作")
print("=" * 60)

# 单列分组
grouped = df.groupby('部门')
print(f"\n分组对象类型: {type(grouped)}")
print(f"分组键: {grouped.groups.keys()}")
print(f"各组大小:\n{grouped.size()}")

# 分组后计算均值
print(f"\n按部门分组求均值:\n{grouped[['薪资', '绩效评分']].mean().round(2)}")

# 分组后计算多种统计量
print(f"\n按部门分组求薪资统计:\n{grouped['薪资'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)}")

print("\n" + "=" * 60)
print("三、agg() 聚合函数")
print("=" * 60)

# 多个聚合函数
print("\n对单列应用多个聚合函数:")
print(df.groupby('部门')['薪资'].agg(['mean', 'median', 'std', 'count']).round(2))

# 对不同列应用不同聚合函数
print("\n对不同列应用不同聚合函数:")
print(df.groupby('部门').agg({
    '薪资': ['mean', 'max'],
    '绩效评分': ['mean', 'min'],
    '工龄(年)': 'mean'
}).round(2))

# 使用自定义函数
print("\n使用自定义函数 (极差):")
print(df.groupby('部门')['薪资'].agg(lambda x: x.max() - x.min()))

print("\n" + "=" * 60)
print("四、命名聚合 (Named Aggregation)")
print("=" * 60)

result = df.groupby('部门').agg(
    平均薪资=('薪资', 'mean'),
    最高薪资=('薪资', 'max'),
    最低薪资=('薪资', 'min'),
    平均绩效=('绩效评分', 'mean'),
    人数=('姓名', 'count')
)
print("\n命名聚合结果:")
print(result.round(2))

print("\n" + "=" * 60)
print("五、transform() 方法")
print("=" * 60)

# transform 返回与原 DataFrame 相同形状的结果
df['部门平均薪资'] = df.groupby('部门')['薪资'].transform('mean').round(2)
df['薪资偏差'] = (df['薪资'] - df['部门平均薪资']).round(2)
print("\ntransform 计算部门平均薪资和偏差:")
print(df[['姓名', '部门', '薪资', '部门平均薪资', '薪资偏差']].to_string())

# transform 标准化
df['薪资标准化'] = df.groupby('部门')['薪资'].transform(lambda x: ((x - x.mean()) / x.std()).round(2))
print("\ntransform 标准化:")
print(df[['姓名', '部门', '薪资', '薪资标准化']].to_string())

# 清理临时列
df = df.drop(columns=['部门平均薪资', '薪资偏差', '薪资标准化'])

print("\n" + "=" * 60)
print("六、filter() 方法")
print("=" * 60)

# 过滤分组: 只保留平均薪资大于15000的部门
filtered = df.groupby('部门').filter(lambda x: x['薪资'].mean() > 15000)
print("\n过滤: 只保留平均薪资>15000的部门:")
print(filtered[['姓名', '部门', '薪资']].to_string())
print(f"\n过滤后部门: {filtered['部门'].unique()}")

# 过滤: 只保留人数>=5的部门
filtered_count = df.groupby('部门').filter(lambda x: len(x) >= 5)
print(f"\n过滤: 只保留人数>=5的部门: {filtered_count['部门'].unique()}")

print("\n" + "=" * 60)
print("七、apply() 方法")
print("=" * 60)

# apply 对每个分组应用自定义函数
def top_n(df_group, n=2, column='薪资'):
    return df_group.nlargest(n, column)

print("\n每个部门薪资最高的2人:")
print(df.groupby('部门').apply(top_n, n=2, column='薪资', include_groups=False).to_string())

# apply 计算每组的薪资占比
def salary_ratio(group):
    group['薪资占比'] = (group['薪资'] / group['薪资'].sum() * 100).round(2)
    return group

result_apply = df.groupby('部门').apply(salary_ratio, include_groups=False)
print("\n每组内薪资占比:")
print(result_apply[['姓名', '薪资', '薪资占比']].to_string())

print("\n" + "=" * 60)
print("八、多列分组")
print("=" * 60)

multi_group = df.groupby(['部门', '职级'])['薪资'].agg(['mean', 'count']).round(2)
print("\n按部门和职级分组:")
print(multi_group.to_string())

# as_index=False 保持分组键为列
multi_group_col = df.groupby(['部门', '职级'], as_index=False)['薪资'].mean().round(2)
print("\nas_index=False:")
print(multi_group_col.to_string())

# reset_index 也可以达到同样效果
multi_group_reset = df.groupby(['部门', '职级'])['薪资'].mean().round(2).reset_index()
print("\nreset_index():")
print(multi_group_reset.to_string())

print("\n" + "=" * 60)
print("九、分组迭代")
print("=" * 60)

print("\n遍历分组结果:")
for name, group in df.groupby('部门'):
    print(f"\n部门: {name} (共{len(group)}人)")
    print(f"  平均薪资: {group['薪资'].mean():.0f}, 平均绩效: {group['绩效评分'].mean():.1f}")

# 获取特定分组
tech_group = df.groupby('部门').get_group('技术部')
print(f"\nget_group('技术部'):\n{tech_group.to_string()}")
