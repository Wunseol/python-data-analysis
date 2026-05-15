# 数据来源: seaborn 内置 tips 数据集
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 7. 分组对比分析
# 分组分析用于比较不同类别之间的差异和规律
# ============================================================

df = sns.load_dataset('tips')
print("=" * 60)
print("小费数据集 - 分组对比分析")
print("=" * 60)

# --------------------------------------------------
# 7.1 groupby() - 分组基础操作
# 按一个或多个变量分组,计算各组的统计量
# --------------------------------------------------
print("\n【7.1 groupby() - 基本分组统计】")
group_sex = df.groupby('sex')['total_bill'].mean()
print("各性别平均账单金额:")
print(group_sex.round(2))

group_day = df.groupby('day')['total_bill'].mean().sort_values(ascending=False)
print("\n各天平均账单金额:")
print(group_day.round(2))

# 多列分组
group_sex_time = df.groupby(['sex', 'time'])['total_bill'].mean()
print("\n性别×用餐时段 平均账单:")
print(group_sex_time.round(2))

# --------------------------------------------------
# 7.2 agg() - 多函数聚合
# 同时计算多个统计量
# --------------------------------------------------
print("\n【7.2 agg() - 多函数聚合】")
agg_result = df.groupby('day').agg(
    total_bill_mean=('total_bill', 'mean'),
    total_bill_median=('total_bill', 'median'),
    total_bill_std=('total_bill', 'std'),
    tip_mean=('tip', 'mean'),
    tip_pct=('tip', lambda x: (x / df.loc[x.index, 'total_bill'] * 100).mean()),
    count=('total_bill', 'count')
)
print(agg_result.round(2))

# 多列多函数
print("\n多列多函数聚合:")
multi_agg = df.groupby('time')[['total_bill', 'tip']].agg(['mean', 'std', 'min', 'max'])
print(multi_agg.round(2))

# --------------------------------------------------
# 7.3 transform() - 分组变换
# 在原 DataFrame 上添加分组统计列,保持原始形状
# --------------------------------------------------
print("\n【7.3 transform() - 分组变换】")
df['day_avg_bill'] = df.groupby('day')['total_bill'].transform('mean')
df['bill_diff_from_day_avg'] = df['total_bill'] - df['day_avg_bill']
print("每日账单与当日均值的差异(前10行):")
print(df[['day', 'total_bill', 'day_avg_bill', 'bill_diff_from_day_avg']].head(10).round(2))

# --------------------------------------------------
# 7.4 filter() - 分组过滤
# 根据组级条件筛选整个组
# --------------------------------------------------
print("\n【7.4 filter() - 分组过滤】")
print(f"过滤前数据量: {len(df)} 行")
filtered = df.groupby('day').filter(lambda x: len(x) > 50)
print(f"过滤后数据量(仅保留样本量>50的天): {len(filtered)} 行")
print(f"保留的天: {filtered['day'].unique()}")

# --------------------------------------------------
# 7.5 apply() - 自定义分组操作
# 使用自定义函数进行灵活的分组计算
# --------------------------------------------------
print("\n【7.5 apply() - 自定义分组操作】")

def top_spender(group, n=3):
    return group.nlargest(n, 'total_bill')[['sex', 'total_bill', 'tip']]

top_by_day = df.groupby('day').apply(top_spender, include_groups=False)
print("每天消费最高的3位顾客:")
print(top_by_day.round(2))

# --------------------------------------------------
# 7.6 对比条形图
# 可视化不同组别的统计差异
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

day_order = ['Thur', 'Fri', 'Sat', 'Sun']

sns.barplot(data=df, x='day', y='total_bill', ax=axes[0, 0],
            order=day_order, hue='day', palette='Set2', legend=False)
axes[0, 0].set_title('各天平均账单金额')

sns.barplot(data=df, x='day', y='tip', ax=axes[0, 1],
            order=day_order, hue='day', palette='Set3', legend=False)
axes[0, 1].set_title('各天平均小费金额')

sns.barplot(data=df, x='sex', y='total_bill', ax=axes[0, 2],
            hue='time', palette='Pastel1')
axes[0, 2].set_title('性别×时段 平均账单')

sns.barplot(data=df, x='day', y='total_bill', ax=axes[1, 0],
            order=day_order, hue='smoker', palette='coolwarm')
axes[1, 0].set_title('各天吸烟/非吸烟顾客账单')

sns.barplot(data=df, x='size', y='total_bill', ax=axes[1, 1],
            hue='size', palette='viridis', legend=False)
axes[1, 1].set_title('用餐人数与账单金额')

sns.boxplot(data=df, x='day', y='tip', ax=axes[1, 2],
            order=day_order, hue='time', palette='Set1')
axes[1, 2].set_title('各天×时段 小费箱线图')

plt.tight_layout()

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / '7_分组对比分析.png', dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------
# 补充: 分组统计汇总表
# --------------------------------------------------
print("\n【补充: 综合分组统计】")
summary = df.groupby(['sex', 'time', 'smoker']).agg(
    avg_bill=('total_bill', 'mean'),
    avg_tip=('tip', 'mean'),
    count=('total_bill', 'count')
).round(2)
print(summary)

print(f"\n图表已保存至: {output_dir}")
print("\n" + "=" * 60)
print("分组对比分析完成! groupby + agg 是数据分析中最常用的组合操作。")
print("=" * 60)
