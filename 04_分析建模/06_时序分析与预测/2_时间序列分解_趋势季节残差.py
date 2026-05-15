# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据（含趋势和季节性成分）
# 本脚本演示时间序列分解: 将序列拆分为趋势、季节性和残差成分

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 构建含趋势和季节性的时间序列
# ============================================================
print("=" * 60)
print("1. 构建含趋势和季节性的模拟时间序列")
print("=" * 60)

np.random.seed(42)
n_periods = 120

# 趋势成分: 线性增长
trend = np.linspace(10, 50, n_periods)

# 季节成分: 周期为12的正弦波
seasonal = 8 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)

# 残差/噪声成分
residual = np.random.randn(n_periods) * 2

# 加法模型: Y = 趋势 + 季节 + 残差
ts_additive = trend + seasonal + residual

# 乘法模型: Y = 趋势 * 季节 * 残差
seasonal_mult = 1 + 0.15 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)
residual_mult = 1 + np.random.randn(n_periods) * 0.03
ts_multiplicative = trend * seasonal_mult * residual_mult

dates = pd.date_range('2020-01-01', periods=n_periods, freq='ME')
ts_add_series = pd.Series(ts_additive, index=dates, name='加法序列')
ts_mul_series = pd.Series(ts_multiplicative, index=dates, name='乘法序列')

print("加法序列 (前5个):")
print(ts_add_series.head())
print(f"\n乘法序列 (前5个):")
print(ts_mul_series.head())

# ============================================================
# 2. 加法模型分解 (model='additive')
# ============================================================
print("\n" + "=" * 60)
print("2. 加法模型分解 (model='additive')")
print("=" * 60)

# seasonal_decompose 参数说明:
# - x: 时间序列 (需有 DatetimeIndex 或 PeriodIndex)
# - model: 'additive'(加法) 或 'multiplicative'(乘法)
# - period: 季节周期长度 (自动推断或手动指定)
# - extrapolate_trend: 是否外推趋势以覆盖两端缺失值

decomp_add = seasonal_decompose(ts_add_series, model='additive', period=12)

print("趋势成分 (前10个):")
print(decomp_add.trend.head(10))

print("\n季节成分 (前12个, 一个完整周期):")
print(decomp_add.seasonal.head(12))

print("\n残差成分 (前10个):")
print(decomp_add.resid.head(10))

# 验证: 加法模型下 观测值 = 趋势 + 季节 + 残差
reconstructed = decomp_add.trend + decomp_add.seasonal + decomp_add.resid
print("\n验证: 观测值 == 趋势 + 季节 + 残差 (去除NaN后):")
valid_idx = decomp_add.trend.notna()
print(np.allclose(ts_add_series[valid_idx], reconstructed[valid_idx]))

# ============================================================
# 3. 乘法模型分解 (model='multiplicative')
# ============================================================
print("\n" + "=" * 60)
print("3. 乘法模型分解 (model='multiplicative')")
print("=" * 60)

decomp_mul = seasonal_decompose(ts_mul_series, model='multiplicative', period=12)

print("趋势成分 (前10个):")
print(decomp_mul.trend.head(10))

print("\n季节成分 (前12个):")
print(decomp_mul.seasonal.head(12))

print("\n残差成分 (前10个):")
print(decomp_mul.resid.head(10))

# 验证: 乘法模型下 观测值 = 趋势 * 季节 * 残差
reconstructed_mul = decomp_mul.trend * decomp_mul.seasonal * decomp_mul.resid
valid_idx_mul = decomp_mul.trend.notna()
print("\n验证: 观测值 == 趋势 * 季节 * 残差 (去除NaN后):")
print(np.allclose(ts_mul_series[valid_idx_mul], reconstructed_mul[valid_idx_mul]))

# ============================================================
# 4. 加法 vs 乘法模型的选择
# ============================================================
print("\n" + "=" * 60)
print("4. 加法 vs 乘法模型的选择原则")
print("=" * 60)
print("""
加法模型 (Additive): Y = Trend + Seasonal + Residual
  - 季节波动的幅度不随趋势水平变化
  - 适用于季节波动相对恒定的序列

乘法模型 (Multiplicative): Y = Trend × Seasonal × Residual
  - 季节波动的幅度随趋势水平增大而增大
  - 适用于季节波动与趋势成正比的序列

选择方法:
  1. 观察时间序列图: 季节波动幅度是否随水平增大
  2. 如果不确定，两种都尝试，比较残差的稳定性
""")

# ============================================================
# 5. 绘制加法分解结果
# ============================================================
fig = decomp_add.plot()
fig.set_size_inches(12, 10)
fig.suptitle('加法模型分解结果', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/2_加法分解.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. 绘制乘法分解结果
# ============================================================
fig = decomp_mul.plot()
fig.set_size_inches(12, 10)
fig.suptitle('乘法模型分解结果', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/2_乘法分解.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. 自定义分解可视化 (更灵活的绘图方式)
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(ts_add_series, label='原始序列', color='steelblue')
axes[0].set_ylabel('原始')
axes[0].legend(loc='upper left')

axes[1].plot(decomp_add.trend, label='趋势', color='red', linewidth=2)
axes[1].set_ylabel('趋势')
axes[1].legend(loc='upper left')

axes[2].plot(decomp_add.seasonal, label='季节', color='green')
axes[2].set_ylabel('季节')
axes[2].legend(loc='upper left')

axes[3].plot(decomp_add.resid, label='残差', color='orange', marker='o', markersize=2)
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[3].set_ylabel('残差')
axes[3].legend(loc='upper left')

fig.suptitle('时间序列分解 — 加法模型 (自定义绘图)', fontsize=14)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/2_自定义分解.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. 提取分解成分进行进一步分析
# ============================================================
print("=" * 60)
print("8. 分解成分的统计分析")
print("=" * 60)

print("趋势成分统计:")
print(decomp_add.trend.describe())

print("\n季节成分统计 (一个周期):")
seasonal_one_cycle = decomp_add.seasonal.iloc[:12]
print(seasonal_one_cycle)
print(f"季节振幅 (最大-最小): {seasonal_one_cycle.max() - seasonal_one_cycle.min():.2f}")

print("\n残差成分统计:")
resid_clean = decomp_add.resid.dropna()
print(resid_clean.describe())
print(f"残差标准差: {resid_clean.std():.4f}")

print("\n图表已保存。")
