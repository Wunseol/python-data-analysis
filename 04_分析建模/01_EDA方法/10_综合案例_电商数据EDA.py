# 数据来源: 自构建电商交易数据
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 10. 综合案例: 电商数据 EDA
# 完整的 EDA 流程演示,从数据创建到洞察生成
# ============================================================

np.random.seed(42)
print("=" * 60)
print("综合案例: 电商交易数据完整 EDA")
print("=" * 60)

# --------------------------------------------------
# 步骤0: 创建示例电商数据
# --------------------------------------------------
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

n_records = 2000
categories = ['电子产品', '服装鞋帽', '食品饮料', '家居用品', '美妆护肤', '图书文具']
cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '南京']
payment_methods = ['支付宝', '微信支付', '银行卡', '货到付款']
platforms = ['APP', 'PC网页', '小程序', 'H5']
genders = ['男', '女']
vip_levels = ['普通会员', '银卡会员', '金卡会员', '钻石会员']

dates = pd.date_range('2023-01-01', '2024-12-31', freq='h')
sampled_dates = np.random.choice(dates, n_records, replace=True)

data = {
    'order_id': [f'ORD{20230001 + i}' for i in range(n_records)],
    'order_time': sampled_dates,
    'category': np.random.choice(categories, n_records, p=[0.25, 0.20, 0.15, 0.15, 0.13, 0.12]),
    'city': np.random.choice(cities, n_records),
    'payment': np.random.choice(payment_methods, n_records, p=[0.35, 0.35, 0.20, 0.10]),
    'platform': np.random.choice(platforms, n_records, p=[0.45, 0.20, 0.25, 0.10]),
    'gender': np.random.choice(genders, n_records, p=[0.45, 0.55]),
    'vip_level': np.random.choice(vip_levels, n_records, p=[0.40, 0.30, 0.20, 0.10]),
    'age': np.clip(np.random.normal(32, 10, n_records).astype(int), 18, 65),
}

base_prices = {'电子产品': 2000, '服装鞋帽': 300, '食品饮料': 80, '家居用品': 500, '美妆护肤': 250, '图书文具': 60}
data['quantity'] = np.random.choice(range(1, 6), n_records, p=[0.30, 0.25, 0.20, 0.15, 0.10])
data['unit_price'] = [max(10, round(base_prices[cat] * np.random.lognormal(0, 0.5))) for cat in data['category']]
data['amount'] = [q * p for q, p in zip(data['quantity'], data['unit_price'])]

is_discount = np.random.random(n_records) < 0.3
discount_rates = np.where(is_discount, np.random.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3], n_records), 0)
data['discount_rate'] = discount_rates
data['actual_amount'] = [round(a * (1 - d), 2) for a, d in zip(data['amount'], data['discount_rate'])]

is_returned = np.random.random(n_records) < 0.08
data['is_returned'] = is_returned.astype(int)

missing_age_idx = np.random.choice(n_records, int(n_records * 0.03), replace=False)
data['age'] = np.array(data['age'], dtype=float)
data['age'][missing_age_idx] = np.nan

missing_city_idx = np.random.choice(n_records, int(n_records * 0.02), replace=False)
data['city'] = np.array(data['city'], dtype=object)
data['city'][missing_city_idx] = np.nan

df = pd.DataFrame(data)

csv_path = output_dir / 'ecommerce_sample.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"示例数据已保存至: {csv_path}")

# ==========================================================
# 正式 EDA 开始
# ==========================================================
df = pd.read_csv(csv_path, parse_dates=['order_time'])
print(f"\n数据加载完成: {len(df)} 条记录, {len(df.columns)} 个字段\n")

# --------------------------------------------------
# 步骤1: 数据概览
# --------------------------------------------------
print("=" * 50)
print("步骤1: 数据概览")
print("=" * 50)
print(f"\n数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n前5行数据:")
print(df.head())
print(f"\n数据类型:")
print(df.dtypes)
print(f"\n描述性统计(数值列):")
print(df.describe().round(2))
print(f"\n描述性统计(分类列):")
print(df.describe(include='object'))

# --------------------------------------------------
# 步骤2: 缺失值分析
# --------------------------------------------------
print("\n" + "=" * 50)
print("步骤2: 缺失值分析")
print("=" * 50)
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0]
print(missing_df)
print(f"\n总缺失值: {df.isnull().sum().sum()}")
print(f"缺失比例: {df.isnull().mean().mean() * 100:.2f}%")

# --------------------------------------------------
# 步骤3: 分布分析
# --------------------------------------------------
print("\n" + "=" * 50)
print("步骤3: 分布分析")
print("=" * 50)
print("\n数值变量分布:")
for col in ['age', 'quantity', 'unit_price', 'amount', 'actual_amount']:
    s = df[col].dropna()
    print(f"\n  {col}:")
    print(f"    均值={s.mean():.2f}, 中位数={s.median():.2f}, 标准差={s.std():.2f}")
    print(f"    偏度={s.skew():.4f}, 峰度={s.kurtosis():.4f}")
    print(f"    范围=[{s.min():.2f}, {s.max():.2f}]")

print("\n分类变量分布:")
for col in ['category', 'city', 'payment', 'platform', 'gender', 'vip_level']:
    print(f"\n  {col}:")
    vc = df[col].value_counts()
    for val, cnt in vc.items():
        print(f"    {val}: {cnt} ({cnt / len(df) * 100:.1f}%)")

# --------------------------------------------------
# 步骤4: 相关性分析
# --------------------------------------------------
print("\n" + "=" * 50)
print("步骤4: 相关性分析")
print("=" * 50)
numeric_cols = ['age', 'quantity', 'unit_price', 'amount', 'actual_amount', 'discount_rate', 'is_returned']
corr = df[numeric_cols].corr()
print("\n相关系数矩阵:")
print(corr.round(3))

high_corr = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            high_corr.append((corr.columns[i], corr.columns[j], r))
if high_corr:
    print("\n强相关变量对 (|r| > 0.5):")
    for v1, v2, r in high_corr:
        print(f"  {v1} ↔ {v2}: r = {r:.4f}")
else:
    print("\n未发现 |r| > 0.5 的强相关变量对")

# --------------------------------------------------
# 步骤5: 时间趋势分析
# --------------------------------------------------
print("\n" + "=" * 50)
print("步骤5: 时间趋势分析")
print("=" * 50)
df['order_month'] = df['order_time'].dt.to_period('M')
df['order_hour'] = df['order_time'].dt.hour
df['order_weekday'] = df['order_time'].dt.day_name()

monthly_stats = df.groupby('order_month')['actual_amount'].agg(['sum', 'mean', 'count'])
print("\n月度销售统计:")
print(monthly_stats.round(2))

hourly_orders = df.groupby('order_hour').size()
print("\n各时段订单量:")
print(hourly_orders)

weekday_orders = df.groupby('order_weekday').size()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_orders = weekday_orders.reindex(weekday_order)
print("\n各星期订单量:")
print(weekday_orders)

# --------------------------------------------------
# 步骤6: 分组对比分析
# --------------------------------------------------
print("\n" + "=" * 50)
print("步骤6: 分组对比分析")
print("=" * 50)

cat_stats = df.groupby('category').agg(
    订单数=('order_id', 'count'),
    总销售额=('actual_amount', 'sum'),
    平均客单价=('actual_amount', 'mean'),
    平均折扣率=('discount_rate', 'mean'),
    退货率=('is_returned', 'mean')
).round(2)
cat_stats['退货率'] = (cat_stats['退货率'] * 100).round(2).astype(str) + '%'
cat_stats['平均折扣率'] = (cat_stats['平均折扣率'] * 100).round(2).astype(str) + '%'
print("\n各品类销售统计:")
print(cat_stats)

vip_stats = df.groupby('vip_level').agg(
    订单数=('order_id', 'count'),
    平均客单价=('actual_amount', 'mean'),
    退货率=('is_returned', 'mean')
).round(2)
vip_stats['退货率'] = (vip_stats['退货率'] * 100).round(2).astype(str) + '%'
print("\n各会员等级统计:")
print(vip_stats)

# --------------------------------------------------
# 步骤7: 可视化
# --------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 7.1 品类销售额
cat_sales = df.groupby('category')['actual_amount'].sum().sort_values(ascending=True)
axes[0, 0].barh(cat_sales.index, cat_sales.values, color='steelblue')
axes[0, 0].set_title('各品类总销售额')
axes[0, 0].set_xlabel('销售额')

# 7.2 支付方式分布
payment_counts = df['payment'].value_counts()
axes[0, 1].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette('Set2', len(payment_counts)))
axes[0, 1].set_title('支付方式分布')

# 7.3 平台分布
sns.countplot(data=df, x='platform', ax=axes[0, 2], hue='platform', palette='Set3', legend=False)
axes[0, 2].set_title('平台订单分布')

# 7.4 客单价分布
axes[1, 0].hist(df['actual_amount'], bins=50, color='teal', edgecolor='white', alpha=0.7)
axes[1, 0].set_title('实际支付金额分布')
axes[1, 0].set_xlabel('金额')
axes[1, 0].set_ylabel('频数')

# 7.5 年龄分布
axes[1, 1].hist(df['age'].dropna(), bins=30, color='coral', edgecolor='white', alpha=0.7)
axes[1, 1].set_title('用户年龄分布')
axes[1, 1].set_xlabel('年龄')

# 7.6 会员等级×品类 交叉热力图
pivot_vip_cat = df.pivot_table(values='actual_amount', index='vip_level', columns='category', aggfunc='mean')
sns.heatmap(pivot_vip_cat, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 2])
axes[1, 2].set_title('会员等级×品类 平均客单价')

# 7.7 月度销售趋势
monthly_sum = df.groupby('order_month')['actual_amount'].sum()
axes[2, 0].plot(range(len(monthly_sum)), monthly_sum.values, marker='o', markersize=3, color='darkblue')
axes[2, 0].set_title('月度销售额趋势')
axes[2, 0].set_xlabel('月份')
axes[2, 0].set_ylabel('销售额')
axes[2, 0].tick_params(axis='x', rotation=45)

# 7.8 各时段订单量
axes[2, 1].bar(hourly_orders.index, hourly_orders.values, color='mediumpurple')
axes[2, 1].set_title('各时段订单量')
axes[2, 1].set_xlabel('小时')
axes[2, 1].set_ylabel('订单量')

# 7.9 退货率对比
return_by_cat = df.groupby('category')['is_returned'].mean() * 100
axes[2, 2].bar(return_by_cat.index, return_by_cat.values, color='salmon')
axes[2, 2].set_title('各品类退货率(%)')
axes[2, 2].set_ylabel('退货率(%)')
axes[2, 2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(output_dir / '10_电商数据EDA.png', dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------
# 步骤8: 生成洞察总结
# --------------------------------------------------
print("\n" + "=" * 50)
print("步骤8: EDA 洞察总结")
print("=" * 50)

top_category = cat_stats['总销售额'].idxmax()
top_city = df['city'].value_counts().index[0]
top_payment = df['payment'].value_counts().index[0]
top_platform = df['platform'].value_counts().index[0]
avg_order_value = df['actual_amount'].mean()
overall_return_rate = df['is_returned'].mean() * 100

print(f"""
  📊 电商数据 EDA 洞察总结
  ─────────────────────────────────
  1. 数据规模: 共 {len(df)} 条订单, {len(df.columns)} 个字段
  2. 缺失情况: age 缺失 {df['age'].isnull().sum()} 条, city 缺失 {df['city'].isnull().sum()} 条
  3. 销售冠军品类: {top_category}
  4. 平均客单价: ¥{avg_order_value:.2f}
  5. 最热门城市: {top_city}
  6. 主流支付方式: {top_payment}
  7. 主流下单平台: {top_platform}
  8. 整体退货率: {overall_return_rate:.2f}%
  9. 折扣订单占比: {(df['discount_rate'] > 0).mean() * 100:.1f}%
  10. 会员结构: 普通会员占比最高

  💡 建议:
  - 关注高退货率品类,优化商品质量描述
  - 加强 APP 端体验优化(占比最高)
  - 针对高价值会员制定差异化营销策略
  - 关注缺失值处理策略,age 列缺失率约 3%
""")

print(f"图表已保存至: {output_dir / '10_电商数据EDA.png'}")
print(f"示例数据已保存至: {csv_path}")
print("\n" + "=" * 60)
print("综合案例 EDA 完成! 完整的 EDA 流程: 概览→缺失值→分布→相关→趋势→分组→洞察")
print("=" * 60)
