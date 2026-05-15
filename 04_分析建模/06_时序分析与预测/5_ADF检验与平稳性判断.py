# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据
# 本脚本演示 ADF 检验与时间序列平稳性判断

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 平稳性概念
# ============================================================
print("=" * 60)
print("1. 平稳性概念")
print("=" * 60)
print("""
平稳性 (Stationarity):
  严格平稳: 序列的所有统计性质都不随时间变化
  弱平稳 (宽平稳): 满足以下三个条件:
    1. 均值恒定 (不随时间变化)
    2. 方差恒定 (不随时间变化)
    3. 自协方差只与滞后阶数有关, 与时间点无关

时间序列分析中通常指弱平稳。

非平稳序列的常见类型:
  - 带趋势的序列 (均值随时间变化)
  - 带季节性的序列 (均值周期性变化)
  - 方差随时间变化的序列 (异方差)
""")

# ============================================================
# 2. 生成平稳与非平稳序列
# ============================================================
print("=" * 60)
print("2. 生成平稳与非平稳序列")
print("=" * 60)

np.random.seed(42)
n = 200

# 平稳序列: 白噪声
stationary = np.random.randn(n) * 2 + 5

# 非平稳序列: 带趋势 (随机游走)
non_stationary = np.cumsum(np.random.randn(n)) + 50

# 非平稳序列: 带确定性趋势
trend_series = np.linspace(10, 60, n) + np.random.randn(n) * 2

dates = pd.date_range('2020-01-01', periods=n, freq='D')
ts_stationary = pd.Series(stationary, index=dates, name='平稳序列')
ts_non_stationary = pd.Series(non_stationary, index=dates, name='非平稳序列(随机游走)')
ts_trend = pd.Series(trend_series, index=dates, name='非平稳序列(确定性趋势)')

print("平稳序列统计:")
print(f"  前50个均值: {stationary[:50].mean():.2f}")
print(f"  后50个均值: {stationary[-50:].mean():.2f}")
print(f"  前50个标准差: {stationary[:50].std():.2f}")
print(f"  后50个标准差: {stationary[-50:].std():.2f}")

print("\n非平稳序列(随机游走)统计:")
print(f"  前50个均值: {non_stationary[:50].mean():.2f}")
print(f"  后50个均值: {non_stationary[-50:].mean():.2f}")
print(f"  前50个标准差: {non_stationary[:50].std():.2f}")
print(f"  后50个标准差: {non_stationary[-50:].std():.2f}")

# ============================================================
# 3. ADF 检验 (Augmented Dickey-Fuller Test)
# ============================================================
print("\n" + "=" * 60)
print("3. ADF 检验")
print("=" * 60)

print("""
ADF 检验 (增广迪基-福勒检验):
  原假设 H0: 序列存在单位根 (非平稳)
  备择假设 H1: 序列不存在单位根 (平稳)

  判断规则:
    - p-value < 0.05: 拒绝原假设, 认为序列平稳
    - p-value >= 0.05: 不能拒绝原假设, 认为序列非平稳

  ADF 统计量越小 (越负), 越倾向于拒绝原假设

  adfuller() 返回值:
    0. adf: ADF 统计量
    1. pvalue: p 值
    2. usedlag: 使用的滞后阶数
    3. nobs: 用于回归的观测数
    4. critical values: 不同显著性水平的临界值
    5. icbest: 信息准则值
""")

# 对平稳序列做 ADF 检验
print("--- 平稳序列的 ADF 检验 ---")
result_stat = adfuller(ts_stationary, autolag='AIC')
print(f"ADF 统计量: {result_stat[0]:.4f}")
print(f"p 值: {result_stat[1]:.6f}")
print(f"使用的滞后阶数: {result_stat[2]}")
print(f"观测数: {result_stat[3]}")
print("临界值:")
for key, value in result_stat[4].items():
    print(f"  {key}: {value:.4f}")

if result_stat[1] < 0.05:
    print("结论: p < 0.05, 拒绝原假设, 序列平稳 ✓")
else:
    print("结论: p >= 0.05, 不能拒绝原假设, 序列非平稳 ✗")

# 对非平稳序列做 ADF 检验
print("\n--- 非平稳序列(随机游走)的 ADF 检验 ---")
result_nonstat = adfuller(ts_non_stationary, autolag='AIC')
print(f"ADF 统计量: {result_nonstat[0]:.4f}")
print(f"p 值: {result_nonstat[1]:.6f}")
print("临界值:")
for key, value in result_nonstat[4].items():
    print(f"  {key}: {value:.4f}")

if result_nonstat[1] < 0.05:
    print("结论: p < 0.05, 拒绝原假设, 序列平稳 ✓")
else:
    print("结论: p >= 0.05, 不能拒绝原假设, 序列非平稳 ✗")

# 对带趋势序列做 ADF 检验
print("\n--- 非平稳序列(确定性趋势)的 ADF 检验 ---")
result_trend = adfuller(ts_trend, autolag='AIC')
print(f"ADF 统计量: {result_trend[0]:.4f}")
print(f"p 值: {result_trend[1]:.6f}")

if result_trend[1] < 0.05:
    print("结论: p < 0.05, 拒绝原假设, 序列平稳 ✓")
else:
    print("结论: p >= 0.05, 不能拒绝原假设, 序列非平稳 ✗")

# ============================================================
# 4. 差分实现平稳化
# ============================================================
print("\n" + "=" * 60)
print("4. 差分实现平稳化")
print("=" * 60)

# 一阶差分
ts_nonstat_diff1 = ts_non_stationary.diff().dropna()
ts_trend_diff1 = ts_trend.diff().dropna()

print("--- 随机游走一阶差分后的 ADF 检验 ---")
result_diff1 = adfuller(ts_nonstat_diff1, autolag='AIC')
print(f"ADF 统计量: {result_diff1[0]:.4f}")
print(f"p 值: {result_diff1[1]:.6f}")
if result_diff1[1] < 0.05:
    print("结论: 一阶差分后平稳 ✓")
else:
    print("结论: 一阶差分后仍非平稳, 需要二阶差分 ✗")

print("\n--- 确定性趋势一阶差分后的 ADF 检验 ---")
result_trend_diff1 = adfuller(ts_trend_diff1, autolag='AIC')
print(f"ADF 统计量: {result_trend_diff1[0]:.4f}")
print(f"p 值: {result_trend_diff1[1]:.6f}")
if result_trend_diff1[1] < 0.05:
    print("结论: 一阶差分后平稳 ✓")
else:
    print("结论: 一阶差分后仍非平稳 ✗")

# ============================================================
# 5. 封装 ADF 检验函数
# ============================================================
print("\n" + "=" * 60)
print("5. 封装 ADF 检验函数")
print("=" * 60)

def adf_test(series, title='', significance=0.05):
    """执行 ADF 检验并打印结果"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF 检验: {title}")
    print(f"  ADF 统计量: {result[0]:.4f}")
    print(f"  p 值: {result[1]:.6f}")
    print(f"  临界值: 1%={result[4]['1%']:.4f}, 5%={result[4]['5%']:.4f}, 10%={result[4]['10%']:.4f}")
    if result[1] < significance:
        print(f"  结论: p={result[1]:.6f} < {significance}, 序列平稳")
    else:
        print(f"  结论: p={result[1]:.6f} >= {significance}, 序列非平稳")
    print()
    return result[1] < significance

# 测试
adf_test(ts_stationary, '平稳序列')
adf_test(ts_non_stationary, '非平稳序列(随机游走)')
adf_test(ts_nonstat_diff1, '随机游走一阶差分')

# ============================================================
# 6. KPSS 检验概念
# ============================================================
print("=" * 60)
print("6. KPSS 检验概念")
print("=" * 60)
print("""
KPSS 检验 (Kwiatkowski-Phillips-Schmidt-Shin):
  与 ADF 检验互为补充:
    ADF: 原假设为"存在单位根(非平稳)"
    KPSS: 原假设为"序列是平稳的"

  两个检验配合使用:
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │ ADF: 平稳        │ ADF: 非平稳      │
  ├──────────────┼──────────────────┼──────────────────┤
  │ KPSS: 平稳   │ 序列平稳         │ 矛盾, 需进一步分析│
  │ KPSS: 非平稳 │ 趋势平稳         │ 序列非平稳        │
  └──────────────┴──────────────────┴──────────────────┘

  趋势平稳: 去除确定性趋势后平稳 (差分不是唯一方法)

  statsmodels 中的 KPSS:
    from statsmodels.tsa.stattools import kpss
    kpss_stat, p_value, lags, crit = kpss(series, regression='c')
    # regression='c' 检验水平平稳, 'ct' 检验趋势平稳
""")

# 实际运行 KPSS
from statsmodels.tsa.stattools import kpss

print("--- KPSS 检验: 平稳序列 ---")
kpss_stat, kpss_pval, kpss_lags, kpss_crit = kpss(ts_stationary, regression='c')
print(f"  KPSS 统计量: {kpss_stat:.4f}")
print(f"  p 值: {kpss_pval:.6f}")
if kpss_pval > 0.05:
    print("  结论: p > 0.05, 不能拒绝平稳性假设, 序列平稳 ✓")
else:
    print("  结论: p <= 0.05, 拒绝平稳性假设, 序列非平稳 ✗")

print("\n--- KPSS 检验: 非平稳序列(随机游走) ---")
kpss_stat2, kpss_pval2, kpss_lags2, kpss_crit2 = kpss(ts_non_stationary, regression='c')
print(f"  KPSS 统计量: {kpss_stat2:.4f}")
print(f"  p 值: {kpss_pval2:.6f}")
if kpss_pval2 > 0.05:
    print("  结论: p > 0.05, 不能拒绝平稳性假设, 序列平稳 ✓")
else:
    print("  结论: p <= 0.05, 拒绝平稳性假设, 序列非平稳 ✗")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 原始序列
ts_stationary.plot(ax=axes[0, 0], title='平稳序列 (白噪声)', color='steelblue')
ts_non_stationary.plot(ax=axes[0, 1], title='非平稳序列 (随机游走)', color='coral')

# 差分后
ts_nonstat_diff1.plot(ax=axes[1, 0], title='随机游走一阶差分', color='green')
axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

ts_trend.plot(ax=axes[1, 1], title='确定性趋势序列', color='purple')
ts_trend_diff1.plot(ax=axes[2, 0], title='确定性趋势一阶差分', color='purple')
axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# ADF p值对比
labels = ['平稳', '随机游走', '确定性趋势', '随机游走\n一阶差分', '趋势\n一阶差分']
p_values = [
    result_stat[1], result_nonstat[1], result_trend[1],
    result_diff1[1], result_trend_diff1[1]
]
colors = ['green' if p < 0.05 else 'red' for p in p_values]
axes[2, 1].bar(labels, p_values, color=colors, alpha=0.7)
axes[2, 1].axhline(y=0.05, color='black', linestyle='--', label='α=0.05')
axes[2, 1].set_title('ADF 检验 p 值对比 (绿色=平稳, 红色=非平稳)')
axes[2, 1].set_ylabel('p 值')
axes[2, 1].legend()
axes[2, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/5_ADF检验.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存。")
