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
# 4. 多变量相关性分析
# 多变量分析用于同时探索多个变量之间的相关关系
# ============================================================

df = sns.load_dataset('iris')
print("=" * 60)
print("鸢尾花数据集 - 多变量相关性分析")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns

# --------------------------------------------------
# 4.1 df.corr() - 相关系数矩阵
# 计算所有数值变量两两之间的 Pearson 相关系数
# --------------------------------------------------
print("\n【4.1 Pearson 相关系数矩阵】")
corr_matrix = df[numeric_cols].corr()
print(corr_matrix.round(4))

# --------------------------------------------------
# 4.2 seaborn heatmap - 相关性热力图
# 用颜色深浅直观展示相关性的强弱
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, ax=axes[0],
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Pearson 相关系数热力图')

# Spearman 秩相关热力图
spearman_corr = df[numeric_cols].corr(method='spearman')
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, ax=axes[1],
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[1].set_title('Spearman 秩相关热力图')

# --------------------------------------------------
# 4.3 seaborn pairplot - 散点图矩阵
# 同时展示所有变量两两之间的散点图和单变量分布
# --------------------------------------------------
pairplot = sns.pairplot(df, hue='species', diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 30},
                        diag_kws={'alpha': 0.6})
pairplot.fig.suptitle('鸢尾花数据集 - 散点图矩阵', y=1.02)

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)
pairplot.fig.savefig(output_dir / '4_散点图矩阵.png', dpi=150, bbox_inches='tight')
plt.close()

# --------------------------------------------------
# 4.4 筛选高相关特征
# 找出相关系数绝对值超过阈值的变量对
# --------------------------------------------------
print("\n【4.4 高相关特征对 (|r| > 0.7)】")
threshold = 0.7
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > threshold:
            high_corr_pairs.append({
                '变量1': corr_matrix.columns[i],
                '变量2': corr_matrix.columns[j],
                '相关系数': round(r, 4),
                '关系强度': '强正相关' if r > 0.7 else '强负相关'
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
print(high_corr_df.to_string(index=False))

# --------------------------------------------------
# 4.5 相关性解读
# 相关系数的含义和注意事项
# --------------------------------------------------
print("\n【4.5 相关性解读指南】")
print("  |r| 范围     强度描述")
print("  ─────────────────────")
print("  0.0 - 0.2   极弱相关或无关")
print("  0.2 - 0.4   弱相关")
print("  0.4 - 0.6   中等相关")
print("  0.6 - 0.8   强相关")
print("  0.8 - 1.0   极强相关")
print()
print("  注意事项:")
print("  1. 相关不等于因果")
print("  2. Pearson 只衡量线性关系,非线性关系需用 Spearman/Kendall")
print("  3. 异常值会显著影响相关系数")
print("  4. 分组数据可能出现 Simpson 悖论")

# 各品种内相关性(避免 Simpson 悖论)
print("\n【4.5b 各品种内相关系数矩阵】")
for species in df['species'].unique():
    subset = df[df['species'] == species][numeric_cols]
    sub_corr = subset.corr()
    print(f"\n  {species}:")
    print(sub_corr.round(3).to_string())

# 按品种着色的相关性热力图(使用全部数据)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, ax=axes[2],
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[2].set_title('下三角相关系数热力图')

plt.tight_layout()
plt.savefig(output_dir / '4_相关性热力图.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存至: {output_dir}")
print("\n" + "=" * 60)
print("多变量相关性分析完成! 热力图和散点图矩阵是发现变量关系的核心工具。")
print("=" * 60)
