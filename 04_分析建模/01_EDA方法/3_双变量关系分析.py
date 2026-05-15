# 数据来源: seaborn 内置 iris 数据集
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 3. 双变量关系分析
# 双变量分析用于探索两个变量之间的关系和依赖性
# ============================================================

df = sns.load_dataset('iris')
print("=" * 60)
print("鸢尾花数据集 - 双变量关系分析")
print("=" * 60)

# --------------------------------------------------
# 3.1 散点图 (Scatter Plot)
# 最直观的双变量关系可视化方法
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', ax=axes[0, 0])
axes[0, 0].set_title('花萼长度 vs 花瓣长度')

sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', ax=axes[0, 1])
axes[0, 1].set_title('花萼长度 vs 花萼宽度')

sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', ax=axes[0, 2])
axes[0, 2].set_title('花瓣长度 vs 花瓣宽度')

# --------------------------------------------------
# 3.2 相关系数 (Pearson)
# 衡量两个变量线性关系的强度和方向,范围 [-1, 1]
# --------------------------------------------------
print("\n【3.2 Pearson 相关系数】")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col_a, col_b = numeric_cols[i], numeric_cols[j]
        corr = df[col_a].corr(df[col_b])
        strength = "强" if abs(corr) > 0.7 else "中" if abs(corr) > 0.4 else "弱"
        direction = "正" if corr > 0 else "负"
        print(f"  {col_a} vs {col_b}: r = {corr:.4f} ({direction}{strength}相关)")

# --------------------------------------------------
# 3.3 交叉表 (Cross-tabulation)
# 用于分析两个分类变量之间的关系
# --------------------------------------------------
print("\n【3.3 交叉表分析】")
df['sepal_length_cat'] = pd.cut(df['sepal_length'], bins=3, labels=['短', '中', '长'])
ct = pd.crosstab(df['species'], df['sepal_length_cat'])
print("品种与花萼长度分类的交叉表:")
print(ct)

print("\n行百分比(各品种中不同长度占比):")
print(pd.crosstab(df['species'], df['sepal_length_cat'], normalize='index').round(3))

# --------------------------------------------------
# 3.4 分组统计 (Grouped Statistics)
# 按分类变量分组后计算数值变量的统计量
# --------------------------------------------------
print("\n【3.4 分组统计】")
group_stats = df.groupby('species')[numeric_cols.tolist()].agg(['mean', 'std', 'min', 'max'])
print("各品种的统计描述:")
print(group_stats.round(2))

# --------------------------------------------------
# 3.5 seaborn jointplot - 联合分布图
# 同时展示两个变量的联合分布和各自的边际分布
# --------------------------------------------------
joint = sns.jointplot(data=df, x='petal_length', y='petal_width', hue='species',
                      height=6, ratio=5, marginal_ticks=True)
joint.fig.suptitle('花瓣长度 vs 花瓣宽度 联合分布图', y=1.02)

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)
joint.fig.savefig(output_dir / '3_双变量联合分布图.png', dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------
# 补充: 按品种分别计算相关系数
# --------------------------------------------------
print("\n【补充: 各品种内相关系数】")
for species in df['species'].unique():
    subset = df[df['species'] == species]
    corr = subset['petal_length'].corr(subset['petal_width'])
    print(f"  {species}: 花瓣长度-花瓣宽度 r = {corr:.4f}")

# --------------------------------------------------
# 补充: Spearman 秩相关
# 适用于非线性单调关系,对异常值更鲁棒
# --------------------------------------------------
print("\n【补充: Spearman 秩相关系数】")
spearman_corr = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr(method='spearman')
print(spearman_corr.round(4))

# 散点图矩阵(部分)
sns.scatterplot(data=df, x='sepal_width', y='petal_width', hue='species', ax=axes[1, 0])
axes[1, 0].set_title('花萼宽度 vs 花瓣宽度')

sns.scatterplot(data=df, x='sepal_length', y='petal_width', hue='species', ax=axes[1, 1])
axes[1, 1].set_title('花萼长度 vs 花瓣宽度')

sns.scatterplot(data=df, x='sepal_width', y='petal_length', hue='species', ax=axes[1, 2])
axes[1, 2].set_title('花萼宽度 vs 花瓣长度')

plt.tight_layout()
plt.savefig(output_dir / '3_双变量散点图.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存至: {output_dir}")
print("\n" + "=" * 60)
print("双变量关系分析完成! 散点图和相关系数是发现变量关系的利器。")
print("=" * 60)
