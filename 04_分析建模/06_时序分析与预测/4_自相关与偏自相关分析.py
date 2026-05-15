# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据 (AR(1)和MA(1)过程)
# 本脚本演示自相关函数(ACF)和偏自相关函数(PACF)的分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成不同类型的模拟时间序列
# ============================================================
print("=" * 60)
print("1. 生成不同类型的模拟时间序列")
print("=" * 60)

np.random.seed(42)
n = 300

# AR(1) 过程: y_t = 0.7 * y_{t-1} + epsilon_t
ar1 = np.zeros(n)
for i in range(1, n):
    ar1[i] = 0.7 * ar1[i - 1] + np.random.randn()

# MA(1) 过程: y_t = epsilon_t + 0.5 * epsilon_{t-1}
ma1 = np.zeros(n)
eps = np.random.randn(n)
for i in range(1, n):
    ma1[i] = eps[i] + 0.5 * eps[i - 1]

# AR(2) 过程: y_t = 0.5 * y_{t-1} + 0.3 * y_{t-2} + epsilon_t
ar2 = np.zeros(n)
for i in range(2, n):
    ar2[i] = 0.5 * ar2[i - 1] + 0.3 * ar2[i - 2] + np.random.randn()

# 白噪声
white_noise = np.random.randn(n)

dates = pd.date_range('2020-01-01', periods=n, freq='D')
ts_ar1 = pd.Series(ar1, index=dates, name='AR(1)')
ts_ma1 = pd.Series(ma1, index=dates, name='MA(1)')
ts_ar2 = pd.Series(ar2, index=dates, name='AR(2)')
ts_wn = pd.Series(white_noise, index=dates, name='白噪声')

print("AR(1) 过程 (前5个):")
print(ts_ar1.head())
print("\nMA(1) 过程 (前5个):")
print(ts_ma1.head())

# ============================================================
# 2. ACF — 自相关函数
# ============================================================
print("\n" + "=" * 60)
print("2. ACF 自相关函数")
print("=" * 60)

# acf() 参数说明:
# - x: 时间序列
# - nlags: 计算的最大滞后阶数
# - fft: 是否使用FFT加速计算
# - alpha: 置信区间宽度 (默认None不计算)

acf_ar1 = acf(ts_ar1, nlags=20)
acf_ma1 = acf(ts_ma1, nlags=20)
acf_wn = acf(ts_wn, nlags=20)

print("AR(1) 的 ACF (滞后0-10):")
for i, v in enumerate(acf_ar1[:11]):
    print(f"  滞后{i:2d}: {v:+.4f}")

print("\nMA(1) 的 ACF (滞后0-10):")
for i, v in enumerate(acf_ma1[:11]):
    print(f"  滞后{i:2d}: {v:+.4f}")

print("\n白噪声的 ACF (滞后0-10):")
for i, v in enumerate(acf_wn[:11]):
    print(f"  滞后{i:2d}: {v:+.4f}")

# 带置信区间的 ACF
acf_ar1_ci, confint = acf(ts_ar1, nlags=20, alpha=0.05, reta=False), None
acf_vals, confint = acf(ts_ar1, nlags=20, alpha=0.05)
print("\nAR(1) 的 ACF 带95%置信区间 (滞后1-5):")
for i in range(1, 6):
    print(f"  滞后{i}: ACF={acf_vals[i]:.4f}, CI=[{confint[i][0]:.4f}, {confint[i][1]:.4f}]")

# ============================================================
# 3. PACF — 偏自相关函数
# ============================================================
print("\n" + "=" * 60)
print("3. PACF 偏自相关函数")
print("=" * 60)

# pacf() 参数说明:
# - x: 时间序列
# - nlags: 计算的最大滞后阶数
# - method: 计算方法 ('ywunbiased'默认, 'ywmle', 'ols'等)

pacf_ar1 = pacf(ts_ar1, nlags=20)
pacf_ma1 = pacf(ts_ma1, nlags=20)

print("AR(1) 的 PACF (滞后0-10):")
for i, v in enumerate(pacf_ar1[:11]):
    print(f"  滞后{i:2d}: {v:+.4f}")

print("\nMA(1) 的 PACF (滞后0-10):")
for i, v in enumerate(pacf_ma1[:11]):
    print(f"  滞后{i:2d}: {v:+.4f}")

# ============================================================
# 4. 使用 plot_acf 和 plot_pacf 绘图
# ============================================================
print("\n" + "=" * 60)
print("4. 绘制 ACF 和 PACF 图")
print("=" * 60)

# plot_acf / plot_pacf 参数说明:
# - x: 时间序列
# - lags: 显示的滞后阶数
# - alpha: 置信区间 (0.05 表示 95% CI)
# - title: 图标题

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# AR(1)
plot_acf(ts_ar1, lags=30, ax=axes[0, 0], title='AR(1) — ACF')
plot_pacf(ts_ar1, lags=30, ax=axes[0, 1], title='AR(1) — PACF')

# MA(1)
plot_acf(ts_ma1, lags=30, ax=axes[1, 0], title='MA(1) — ACF')
plot_pacf(ts_ma1, lags=30, ax=axes[1, 1], title='MA(1) — PACF')

# AR(2)
plot_acf(ts_ar2, lags=30, ax=axes[2, 0], title='AR(2) — ACF')
plot_pacf(ts_ar2, lags=30, ax=axes[2, 1], title='AR(2) — PACF')

# 白噪声
plot_acf(ts_wn, lags=30, ax=axes[3, 0], title='白噪声 — ACF')
plot_pacf(ts_wn, lags=30, ax=axes[3, 1], title='白噪声 — PACF')

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/4_ACF_PACF对比.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. 解读 ACF/PACF 图 — 识别 AR/MA 阶数
# ============================================================
print("\n" + "=" * 60)
print("5. 解读 ACF/PACF 图 — 识别 AR/MA 阶数")
print("=" * 60)
print("""
ACF/PACF 图解读规则:

┌────────────┬─────────────────────────┬─────────────────────────┐
│ 模型类型   │ ACF 特征                │ PACF 特征               │
├────────────┼─────────────────────────┼─────────────────────────┤
│ AR(p)      │ 拖尾 (指数衰减)         │ p阶截尾                 │
│ MA(q)      │ q阶截尾                 │ 拖尾 (指数衰减)         │
│ ARMA(p,q)  │ 拖尾                    │ 拖尾                    │
│ 白噪声     │ 全部不显著              │ 全部不显著              │
└────────────┴─────────────────────────┴─────────────────────────┘

"截尾": 在某阶之后突然变为0 (落在置信区间内)
"拖尾": 逐渐衰减, 不突然变为0

具体到本例:
  - AR(1): ACF拖尾(指数衰减), PACF在滞后1后截尾
  - MA(1): ACF在滞后1后截尾, PACF拖尾(指数衰减)
  - AR(2): ACF拖尾, PACF在滞后2后截尾
  - 白噪声: ACF和PACF均在置信区间内
""")

# ============================================================
# 6. 非平稳序列的 ACF/PACF 特征
# ============================================================
print("=" * 60)
print("6. 非平稳序列的 ACF/PACF 特征")
print("=" * 60)

# 带趋势的序列
trend_data = np.cumsum(np.random.randn(n)) + 50
ts_trend = pd.Series(trend_data, index=dates, name='带趋势')

# 差分后
ts_diff = ts_trend.diff().dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_acf(ts_trend, lags=40, ax=axes[0, 0], title='带趋势序列 — ACF (非平稳)')
plot_pacf(ts_trend, lags=40, ax=axes[0, 1], title='带趋势序列 — PACF (非平稳)')

plot_acf(ts_diff, lags=40, ax=axes[1, 0], title='一阶差分后 — ACF (平稳)')
plot_pacf(ts_diff, lags=40, ax=axes[1, 1], title='一阶差分后 — PACF (平稳)')

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/4_非平稳ACF_PACF.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
非平稳序列的 ACF 特征:
  - ACF 衰减非常缓慢 (不趋于0)
  - 这是非平稳性的重要信号
  - 差分后 ACF 应快速衰减
""")

print("图表已保存。")
