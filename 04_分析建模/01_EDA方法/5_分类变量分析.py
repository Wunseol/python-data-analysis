# 数据来源: seaborn 内置 titanic 数据集
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 5. 分类变量分析
# 分类变量分析用于理解离散型变量的分布和变量间的关系
# ============================================================

df = sns.load_dataset('titanic')
print("=" * 60)
print("泰坦尼克号数据集 - 分类变量分析")
print("=" * 60)

# --------------------------------------------------
# 5.1 value_counts() - 频率统计
# 分类变量分析的第一步: 了解各类别的频次分布
# --------------------------------------------------
print("\n【5.1 value_counts() - 频率统计】")
cat_cols = ['survived', 'pclass', 'sex', 'embarked', 'class', 'who', 'alone']
for col in cat_cols:
    print(f"\n  {col} 分布:")
    vc = df[col].value_counts()
    for val, count in vc.items():
        pct = count / len(df) * 100
        print(f"    {val}: {count} 人 ({pct:.1f}%)")

# --------------------------------------------------
# 5.2 seaborn countplot - 计数条形图
# 直观展示各类别的数量
# --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.countplot(data=df, x='survived', ax=axes[0, 0], hue='survived', palette='Set2', legend=False)
axes[0, 0].set_title('存活情况')
axes[0, 0].set_xticklabels(['未存活', '存活'])

sns.countplot(data=df, x='pclass', ax=axes[0, 1], hue='pclass', palette='Set3', legend=False)
axes[0, 1].set_title('船舱等级分布')

sns.countplot(data=df, x='sex', ax=axes[0, 2], hue='sex', palette='Pastel1', legend=False)
axes[0, 2].set_title('性别分布')

sns.countplot(data=df, x='embarked', ax=axes[1, 0], hue='embarked', palette='Set1', legend=False)
axes[1, 0].set_title('登船港口分布')

sns.countplot(data=df, x='class', ax=axes[1, 1], hue='class', palette='husl', legend=False)
axes[1, 1].set_title('船舱等级(有序)分布')

sns.countplot(data=df, x='who', ax=axes[1, 2], hue='who', palette='Paired', legend=False)
axes[1, 2].set_title('乘客类型分布')

plt.tight_layout()

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# 5.3 pd.crosstab() - 交叉表
# 分析两个分类变量之间的关系
# --------------------------------------------------
print("\n【5.3 pd.crosstab() - 交叉表分析】")
ct_sex_survived = pd.crosstab(df['sex'], df['survived'], margins=True)
ct_sex_survived.columns = ['未存活', '存活', '合计']
print("性别与存活情况交叉表:")
print(ct_sex_survived)

print("\n行百分比(各性别存活率):")
ct_pct = pd.crosstab(df['sex'], df['survived'], normalize='index')
ct_pct.columns = ['未存活率', '存活率']
print((ct_pct * 100).round(2))

# --------------------------------------------------
# 5.4 卡方检验概念
# 检验两个分类变量是否独立
# H0: 两个变量相互独立; H1: 两个变量有关联
# --------------------------------------------------
print("\n【5.4 卡方检验】")
from scipy.stats import chi2_contingency

ct_for_test = pd.crosstab(df['sex'], df['survived'])
chi2, p_value, dof, expected = chi2_contingency(ct_for_test)
print(f"  性别 vs 存活:")
print(f"    卡方统计量: {chi2:.4f}")
print(f"    p值: {p_value:.2e}")
print(f"    自由度: {dof}")
print(f"    结论: {'拒绝独立假设(有关联)' if p_value < 0.05 else '不能拒绝独立假设'}")

ct_class_survived = pd.crosstab(df['pclass'], df['survived'])
chi2, p_value, dof, expected = chi2_contingency(ct_class_survived)
print(f"\n  船舱等级 vs 存活:")
print(f"    卡方统计量: {chi2:.4f}")
print(f"    p值: {p_value:.2e}")
print(f"    自由度: {dof}")
print(f"    结论: {'拒绝独立假设(有关联)' if p_value < 0.05 else '不能拒绝独立假设'}")

# --------------------------------------------------
# 5.5 比例分析
# 比较不同组别的存活率等比例指标
# --------------------------------------------------
print("\n【5.5 比例分析 - 各组存活率】")
survival_by_sex = df.groupby('sex')['survived'].mean()
print(f"  男性存活率: {survival_by_sex['male']:.2%}")
print(f"  女性存活率: {survival_by_sex['female']:.2%}")

survival_by_class = df.groupby('pclass')['survived'].mean()
for cls, rate in survival_by_class.items():
    print(f"  {cls}等舱存活率: {rate:.2%}")

survival_by_embarked = df.groupby('embarked')['survived'].mean()
for port, rate in survival_by_embarked.items():
    print(f"  {port}港登船存活率: {rate:.2%}")

# --------------------------------------------------
# 5.6 堆叠条形图
# 展示两个分类变量的联合分布
# --------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

ct_sex = pd.crosstab(df['sex'], df['survived'])
ct_sex.columns = ['未存活', '存活']
ct_sex.plot(kind='bar', stacked=True, ax=axes2[0], color=['#e74c3c', '#2ecc71'])
axes2[0].set_title('性别与存活(堆叠条形图)')
axes2[0].set_xlabel('性别')
axes2[0].set_ylabel('人数')
axes2[0].tick_params(axis='x', rotation=0)

ct_class = pd.crosstab(df['pclass'], df['survived'])
ct_class.columns = ['未存活', '存活']
ct_class.plot(kind='bar', stacked=True, ax=axes2[1], color=['#e74c3c', '#2ecc71'])
axes2[1].set_title('船舱等级与存活(堆叠条形图)')
axes2[1].set_xlabel('船舱等级')
axes2[1].set_ylabel('人数')
axes2[1].tick_params(axis='x', rotation=0)

ct_embarked = pd.crosstab(df['embarked'], df['survived'])
ct_embarked.columns = ['未存活', '存活']
ct_embarked.plot(kind='bar', stacked=True, ax=axes2[2], color=['#e74c3c', '#2ecc71'])
axes2[2].set_title('登船港口与存活(堆叠条形图)')
axes2[2].set_xlabel('登船港口')
axes2[2].set_ylabel('人数')
axes2[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(output_dir / '5_分类变量_计数图.png', dpi=150, bbox_inches='tight')
plt.close()

plt.savefig(output_dir / '5_分类变量_堆叠图.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存至: {output_dir}")
print("\n" + "=" * 60)
print("分类变量分析完成! 交叉表和卡方检验是分析分类变量关联性的核心方法。")
print("=" * 60)
