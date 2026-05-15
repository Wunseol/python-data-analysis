# 数据来源: 脚本内自建销售数据并导出为 CSV
# 本脚本演示综合销售数据分析: 加载 → 清洗 → 分组 → 透视 → 时间序列 → 导出

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("一、生成模拟销售数据")
print("=" * 60)

np.random.seed(2024)

n_records = 500
date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
dates = np.random.choice(date_range, n_records)

regions = ['华东', '华南', '华北', '西南', '华中']
products = ['笔记本电脑', '智能手机', '平板电脑', '智能手表', '耳机']
channels = ['线上', '线下']
salesmen = ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十']

df = pd.DataFrame({
    '订单日期': dates,
    '地区': np.random.choice(regions, n_records),
    '产品名称': np.random.choice(products, n_records),
    '销售渠道': np.random.choice(channels, n_records),
    '销售员': np.random.choice(salesmen, n_records),
    '销售数量': np.random.randint(1, 50, n_records),
    '单价': np.random.choice([3999, 5999, 2999, 1299, 699, 8999, 4999, 1999, 399, 599], n_records)
})

df['销售额'] = df['销售数量'] * df['单价']

# 故意添加一些数据问题
problem_indices = np.random.choice(n_records, 20, replace=False)
df.loc[problem_indices[:5], '销售数量'] = np.nan
df.loc[problem_indices[5:10], '地区'] = np.nan
df.loc[problem_indices[10:15], '单价'] = np.nan
df.loc[problem_indices[15:], '销售员'] = np.nan

# 添加重复行
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

# 保存原始数据
raw_csv_path = OUTPUT_DIR / 'sales_raw.csv'
df.to_csv(raw_csv_path, index=False, encoding='utf-8-sig')
print(f"原始数据已保存: {raw_csv_path}")
print(f"数据形状: {df.shape}")
print(f"\n前5行:\n{df.head()}")

print("\n" + "=" * 60)
print("二、数据加载与初步检查")
print("=" * 60)

df = pd.read_csv(raw_csv_path, encoding='utf-8-sig', parse_dates=['订单日期'])
print(f"数据形状: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n重复行数量: {df.duplicated().sum()}")
print(f"\n基本统计:\n{df.describe().round(2)}")

print("\n" + "=" * 60)
print("三、数据清洗")
print("=" * 60)

# 1. 删除重复行
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"\n删除重复行: {before} → {after} (删除{before - after}行)")

# 2. 处理缺失值
print(f"\n清洗前缺失值:\n{df.isnull().sum()}")

# 地区缺失 - 用众数填充
df['地区'] = df['地区'].fillna(df['地区'].mode()[0])

# 销售数量缺失 - 用中位数填充
df['销售数量'] = df['销售数量'].fillna(df['销售数量'].median()).astype(int)

# 单价缺失 - 用同产品的平均单价填充
df['单价'] = df.groupby('产品名称')['单价'].transform(
    lambda x: x.fillna(x.mean())
).round(2)

# 销售员缺失 - 标记为未知
df['销售员'] = df['销售员'].fillna('未知')

# 重新计算销售额
df['销售额'] = df['销售数量'] * df['单价']

print(f"\n清洗后缺失值:\n{df.isnull().sum()}")
print(f"清洗后数据形状: {df.shape}")

print("\n" + "=" * 60)
print("四、特征工程")
print("=" * 60)

# 提取日期特征
df['年份'] = df['订单日期'].dt.year
df['月份'] = df['订单日期'].dt.month
df['季度'] = df['订单日期'].dt.quarter
df['星期'] = df['订单日期'].dt.dayofweek
df['是否周末'] = df['星期'].isin([5, 6]).map({True: '周末', False: '工作日'})

# 销售额分箱
df['销售额等级'] = pd.qcut(df['销售额'], q=4, labels=['低', '中低', '中高', '高'])

print("新增特征后数据:")
print(df.head().to_string())
print(f"\n各销售额等级分布:\n{df['销售额等级'].value_counts().sort_index()}")

print("\n" + "=" * 60)
print("五、分组聚合分析")
print("=" * 60)

# 1. 按地区统计
region_stats = df.groupby('地区').agg(
    总销售额=('销售额', 'sum'),
    平均销售额=('销售额', 'mean'),
    订单数量=('销售额', 'count'),
    平均数量=('销售数量', 'mean')
).round(2)
region_stats = region_stats.sort_values('总销售额', ascending=False)
print("\n按地区统计:")
print(region_stats.to_string())

# 2. 按产品统计
product_stats = df.groupby('产品名称').agg(
    总销售额=('销售额', 'sum'),
    总销量=('销售数量', 'sum'),
    平均单价=('单价', 'mean'),
    订单数量=('销售额', 'count')
).round(2)
product_stats = product_stats.sort_values('总销售额', ascending=False)
print("\n按产品统计:")
print(product_stats.to_string())

# 3. 按渠道统计
channel_stats = df.groupby('销售渠道').agg(
    总销售额=('销售额', 'sum'),
    订单数量=('销售额', 'count'),
    平均销售额=('销售额', 'mean')
).round(2)
print("\n按渠道统计:")
print(channel_stats.to_string())

# 4. 按销售员统计
salesman_stats = df.groupby('销售员').agg(
    总销售额=('销售额', 'sum'),
    订单数量=('销售额', 'count'),
    平均订单额=('销售额', 'mean')
).round(2)
salesman_stats = salesman_stats.sort_values('总销售额', ascending=False)
print("\n按销售员统计:")
print(salesman_stats.to_string())

print("\n" + "=" * 60)
print("六、透视表分析")
print("=" * 60)

# 1. 地区 × 产品 销售额透视
pt_region_product = pd.pivot_table(
    df, values='销售额', index='地区', columns='产品名称',
    aggfunc='sum', fill_value=0, margins=True, margins_name='合计'
)
print("\n地区 × 产品 销售额透视表:")
print(pt_region_product.round(0).to_string())

# 2. 渠道 × 季度 订单数透视
pt_channel_quarter = pd.pivot_table(
    df, values='订单日期', index='销售渠道', columns='季度',
    aggfunc='count', fill_value=0
)
pt_channel_quarter.columns = [f'Q{q}' for q in pt_channel_quarter.columns]
print("\n渠道 × 季度 订单数透视表:")
print(pt_channel_quarter.to_string())

# 3. 地区 × 渠道 平均销售额
pt_region_channel = pd.pivot_table(
    df, values='销售额', index='地区', columns='销售渠道',
    aggfunc='mean', fill_value=0
).round(2)
print("\n地区 × 渠道 平均销售额:")
print(pt_region_channel.to_string())

print("\n" + "=" * 60)
print("七、时间序列分析")
print("=" * 60)

# 设置日期索引
df_ts = df.set_index('订单日期')

# 1. 月度销售趋势
monthly_sales = df_ts['销售额'].resample('ME').sum()
print("\n月度销售额趋势:")
print(monthly_sales.round(0).to_string())

# 2. 月度环比增长率
monthly_growth = monthly_sales.pct_change().dropna()
print("\n月度环比增长率:")
print((monthly_growth * 100).round(2).to_string())

# 3. 季度汇总
quarterly_sales = df_ts['销售额'].resample('QE').agg(['sum', 'mean', 'count'])
quarterly_sales.columns = ['季度总额', '季度均值', '订单数']
print("\n季度汇总:")
print(quarterly_sales.round(0).to_string())

# 4. 滚动平均
monthly_sales_df = monthly_sales.to_frame('月销售额')
monthly_sales_df['3月均线'] = monthly_sales_df['月销售额'].rolling(window=3).mean()
monthly_sales_df['6月均线'] = monthly_sales_df['月销售额'].rolling(window=6).mean()
print("\n月度销售额与滚动均线:")
print(monthly_sales_df.round(0).to_string())

# 5. 同比分析 (2024 vs 2023)
sales_2023 = df_ts.loc['2023', '销售额'].resample('ME').sum()
sales_2024 = df_ts.loc['2024', '销售额'].resample('ME').sum()

yoy = pd.DataFrame({
    '2023年': sales_2023.values,
    '2024年': sales_2024.values[:len(sales_2023)] if len(sales_2024) < len(sales_2023) else sales_2024.values
}, index=range(1, min(len(sales_2023), len(sales_2024)) + 1))
yoy.index.name = '月份'
yoy['同比增长率(%)'] = ((yoy['2024年'] - yoy['2023年']) / yoy['2023年'] * 100).round(2)
print("\n同比分析 (2023 vs 2024):")
print(yoy.round(0).to_string())

# 6. 周内销售模式
weekday_sales = df_ts.groupby(df_ts.index.dayofweek)['销售额'].mean()
weekday_sales.index = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
print("\n周内日均销售额:")
print(weekday_sales.round(0).to_string())

print("\n" + "=" * 60)
print("八、交叉表分析")
print("=" * 60)

# 地区 × 渠道 频数交叉表
ct_region_channel = pd.crosstab(df['地区'], df['销售渠道'], margins=True, margins_name='合计')
print("\n地区 × 渠道 频数交叉表:")
print(ct_region_channel.to_string())

# 地区 × 渠道 比例交叉表
ct_region_channel_pct = pd.crosstab(df['地区'], df['销售渠道'], normalize='index')
print("\n地区 × 渠道 行比例:")
print((ct_region_channel_pct * 100).round(2).to_string())

print("\n" + "=" * 60)
print("九、关键指标汇总")
print("=" * 60)

total_revenue = df['销售额'].sum()
total_orders = len(df)
avg_order_value = df['销售额'].mean()
top_region = df.groupby('地区')['销售额'].sum().idxmax()
top_product = df.groupby('产品名称')['销售额'].sum().idxmax()
top_salesman = df.groupby('销售员')['销售额'].sum().idxmax()

print(f"\n总销售额: ¥{total_revenue:,.0f}")
print(f"总订单数: {total_orders}")
print(f"平均订单额: ¥{avg_order_value:,.0f}")
print(f"最佳地区: {top_region}")
print(f"最佳产品: {top_product}")
print(f"最佳销售员: {top_salesman}")

print("\n" + "=" * 60)
print("十、导出分析结果")
print("=" * 60)

# 1. 导出清洗后数据
clean_csv_path = OUTPUT_DIR / 'sales_cleaned.csv'
df.to_csv(clean_csv_path, index=False, encoding='utf-8-sig')
print(f"\n清洗后数据已保存: {clean_csv_path}")

# 2. 导出各统计结果到 Excel
result_excel_path = OUTPUT_DIR / 'sales_analysis.xlsx'
try:
    with pd.ExcelWriter(result_excel_path, engine='openpyxl') as writer:
        region_stats.to_excel(writer, sheet_name='地区统计')
        product_stats.to_excel(writer, sheet_name='产品统计')
        channel_stats.to_excel(writer, sheet_name='渠道统计')
        salesman_stats.to_excel(writer, sheet_name='销售员统计')
        pt_region_product.round(0).to_excel(writer, sheet_name='地区产品透视')
        monthly_sales_df.round(0).to_excel(writer, sheet_name='月度趋势')
        yoy.round(0).to_excel(writer, sheet_name='同比分析')
    print(f"分析结果 Excel 已保存: {result_excel_path}")
except ImportError:
    print("未安装 openpyxl，跳过 Excel 导出")

# 3. 导出月度趋势 JSON
trend_json_path = OUTPUT_DIR / 'monthly_trend.json'
monthly_sales_df.round(0).to_json(trend_json_path, orient='index', force_ascii=False)
print(f"月度趋势 JSON 已保存: {trend_json_path}")

# 4. 导出关键指标摘要
summary = pd.DataFrame({
    '指标': ['总销售额', '总订单数', '平均订单额', '最佳地区', '最佳产品', '最佳销售员'],
    '值': [f'¥{total_revenue:,.0f}', str(total_orders), f'¥{avg_order_value:,.0f}',
           top_region, top_product, top_salesman]
})
summary_csv_path = OUTPUT_DIR / 'sales_summary.csv'
summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
print(f"关键指标摘要已保存: {summary_csv_path}")

print(f"\n所有输出文件保存在: {OUTPUT_DIR}")
print("\n分析完成!")
