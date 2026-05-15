# 数据来源: seaborn 内置 diamonds 数据集
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 8. 特征交叉分析
# 特征交叉分析用于探索多个特征组合后对目标变量的影响
# ============================================================

df = sns.load_dataset('diamonds')
print("=" * 60)
print("钻石数据集 - 特征交叉分析")
print("=" * 60)

# --------------------------------------------------
# 8.1 pivot_table() - 透视表
# 类似 Excel 透视表,灵活地按多维度汇总数据
# --------------------------------------------------
print("\n【8.1 pivot_table() - 透视表】")
pivot_price = df.pivot_table(values='price', index='cut', columns='color', aggfunc='mean')
print("切工×颜色 平均价格透视表:")
print(pivot_price.round(0))

pivot_count = df.pivot_table(values='price', index='cut', columns='color', aggfunc='count')
print("\n切工×颜色 数量透视表:")
print(pivot_count)

# 多个聚合函数
pivot_multi = df.pivot_table(values='price', index='cut', columns='color',
                             aggfunc=['mean', 'median', 'std'])
print("\n切工×颜色 多聚合函数透视表(部分):")
print(pivot_multi['mean'].round(0))

# --------------------------------------------------
# 8.2 多级 groupby
# 按多个变量分组,进行更细粒度的分析
# --------------------------------------------------
print("\n【8.2 多级 groupby】")
multi_group = df.groupby(['cut', 'clarity'])['price'].agg(['mean', 'count']).round(2)
print("切工×净度 分组统计(前15行):")
print(multi_group.head(15))

# --------------------------------------------------
# 8.3 交互特征
# 通过特征组合创建新特征,捕捉变量间的交互效应
# --------------------------------------------------
print("\n【8.3 交互特征】")
df['carat_x_depth'] = df['carat'] * df['depth']
df['carat_x_table'] = df['carat'] * df['table']
df['volume_approx'] = df['x'] * df['y'] * df['z']
df['price_per_carat'] = df['price'] / df['carat']

print("新增交互特征:")
print(df[['carat', 'depth', 'carat_x_depth', 'volume_approx', 'price_per_carat']].head(10).round(2))

# 交互特征与价格的相关性
print("\n交互特征与价格的相关系数:")
for feat in ['carat_x_depth', 'carat_x_table', 'volume_approx', 'price_per_carat']:
    valid = df[[feat, 'price']].replace([np.inf, -np.inf], np.nan).dropna()
    corr = valid[feat].corr(valid['price'])
    print(f"  {feat} vs price: r = {corr:.4f}")

# --------------------------------------------------
# 8.4 交叉特征热力图
# 用热力图展示两个分类特征交叉后目标变量的分布
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 13))

sns.heatmap(pivot_price, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 0],
            cbar_kws={'label': '平均价格'})
axes[0, 0].set_title('切工×颜色 平均价格热力图')

pivot_carat = df.pivot_table(values='carat', index='cut', columns='color', aggfunc='mean')
sns.heatmap(pivot_carat, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[0, 1],
            cbar_kws={'label': '平均克拉数'})
axes[0, 1].set_title('切工×颜色 平均克拉数热力图')

# --------------------------------------------------
# 8.5 seaborn catplot - 分类图
# 展示多分类变量与数值变量的关系
# --------------------------------------------------
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

sns.boxplot(data=df, x='cut', y='price', hue='color',
            order=cut_order, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('切工×颜色 价格箱线图')
axes[1, 0].legend(fontsize=7, title='颜色', ncol=2)

sns.barplot(data=df, x='clarity', y='price_per_carat',
            hue='cut', hue_order=cut_order, ax=axes[1, 1], palette='muted')
axes[1, 1].set_title('净度×切工 每克拉价格')
axes[1, 1].legend(fontsize=7, title='切工')

plt.tight_layout()

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / '8_特征交叉分析.png', dpi=150, bbox_inches='tight')
plt.close()

# 使用 catplot 生成更复杂的交叉图
g = sns.catplot(data=df.sample(3000, random_state=42), x='cut', y='price',
                col='color', kind='box', col_wrap=4, height=3,
                order=cut_order, palette='Set2', hue='cut', legend=False)
g.fig.suptitle('各颜色下切工对价格的影响', y=1.02)
g.fig.savefig(output_dir / '8_特征交叉_catplot.png', dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------
# 补充: 交互效应分析
# --------------------------------------------------
print("\n【补充: 交互效应分析】")
overall_mean = df['price'].mean()
for cut in cut_order:
    for color in sorted(df['color'].unique()):
        subset = df[(df['cut'] == cut) & (df['color'] == color)]
        if len(subset) > 0:
            group_mean = subset['price'].mean()
            diff = group_mean - overall_mean
            print(f"  {cut:10s} × {color}: 均价={group_mean:8.0f}, 偏离总均值={diff:+8.0f}")

print(f"\n图表已保存至: {output_dir}")
print("\n" + "=" * 60)
print("特征交叉分析完成! 透视表和热力图是发现特征交互效应的核心工具。")
print("=" * 60)
