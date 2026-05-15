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
# 2. 单变量分布分析
# 单变量分析是 EDA 的基础,用于理解每个变量自身的分布特征
# ============================================================

df = sns.load_dataset('tips')
print("=" * 60)
print("小费数据集 - 单变量分布分析")
print("=" * 60)

# --------------------------------------------------
# 2.1 value_counts() - 频率统计
# 统计分类变量各值出现的次数和比例
# --------------------------------------------------
print("\n【2.1 value_counts() - 频率统计】")
print("星期分布:")
print(df['day'].value_counts())
print("\n星期比例:")
print(df['day'].value_counts(normalize=True).round(3))

# --------------------------------------------------
# 2.2 df.describe() - 描述性统计
# 对数值变量进行全面的统计描述
# --------------------------------------------------
print("\n【2.2 df.describe() - 数值列描述性统计】")
print(df.describe())

# --------------------------------------------------
# 2.3 matplotlib 直方图
# 最基本的分布可视化方法,展示数据的频率分布
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].hist(df['total_bill'], bins=20, color='steelblue', edgecolor='white')
axes[0, 0].set_title('总账单金额直方图 (matplotlib)')
axes[0, 0].set_xlabel('总账单金额')
axes[0, 0].set_ylabel('频数')

# --------------------------------------------------
# 2.4 KDE 核密度估计图
# 用平滑曲线展示数据的概率密度分布
# --------------------------------------------------
from scipy.stats import gaussian_kde

kde = gaussian_kde(df['total_bill'])
x_range = np.linspace(df['total_bill'].min(), df['total_bill'].max(), 200)
axes[0, 1].plot(x_range, kde(x_range), color='darkorange', linewidth=2)
axes[0, 1].fill_between(x_range, kde(x_range), alpha=0.3, color='darkorange')
axes[0, 1].set_title('总账单金额KDE图')
axes[0, 1].set_xlabel('总账单金额')
axes[0, 1].set_ylabel('密度')

# --------------------------------------------------
# 2.5 seaborn histplot + kde
# seaborn 提供更美观的直方图,支持叠加 kde 曲线
# --------------------------------------------------
sns.histplot(df['total_bill'], bins=20, kde=True, ax=axes[0, 2], color='teal')
axes[0, 2].set_title('总账单金额直方图+KDE (seaborn)')

# --------------------------------------------------
# 2.6 箱线图 - 单变量
# 展示中位数、四分位数、异常值等分布特征
# --------------------------------------------------
axes[1, 0].boxplot(df['total_bill'].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[1, 0].set_title('总账单金额箱线图')
axes[1, 0].set_ylabel('总账单金额')

# 小费金额的分布
sns.histplot(df['tip'], bins=20, kde=True, ax=axes[1, 1], color='salmon')
axes[1, 1].set_title('小费金额直方图+KDE')

# 用餐人数分布
sns.countplot(x='size', data=df, ax=axes[1, 2], hue='size', palette='Set2', legend=False)
axes[1, 2].set_title('用餐人数分布')
axes[1, 2].set_xlabel('用餐人数')
axes[1, 2].set_ylabel('频数')

plt.tight_layout()

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / '2_单变量分布分析.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存至: {output_dir / '2_单变量分布分析.png'}")

# --------------------------------------------------
# 2.7 偏度(Skewness)和峰度(Kurtosis)
# 偏度衡量分布的对称性,峰度衡量分布的尖峭程度
# 正态分布: 偏度=0, 峰度=0 (超额峰度)
# --------------------------------------------------
print("\n【2.7 偏度和峰度分析】")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"\n  {col}:")
    print(f"    偏度(Skewness): {skewness:.4f}", end="")
    if skewness > 0.5:
        print(" → 右偏(正偏)", end="")
    elif skewness < -0.5:
        print(" → 左偏(负偏)", end="")
    else:
        print(" → 近似对称", end="")
    print(f"\n    峰度(Kurtosis): {kurtosis:.4f}", end="")
    if kurtosis > 1:
        print(" → 尖峰分布(重尾)")
    elif kurtosis < -1:
        print(" → 扁平分布(轻尾)")
    else:
        print(" → 近似正态")

# --------------------------------------------------
# 补充: 分位数分析
# --------------------------------------------------
print("\n【补充: 分位数分析】")
for col in ['total_bill', 'tip']:
    print(f"\n  {col} 分位数:")
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    q_values = df[col].quantile(quantiles)
    for q, v in q_values.items():
        print(f"    {q*100:5.1f}%: {v:.2f}")

print("\n" + "=" * 60)
print("单变量分布分析完成! 偏度和峰度帮助我们判断数据是否需要变换。")
print("=" * 60)
