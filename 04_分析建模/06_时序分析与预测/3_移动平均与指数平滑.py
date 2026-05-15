# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据
# 本脚本演示移动平均与指数平滑方法

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成模拟时间序列数据
# ============================================================
print("=" * 60)
print("1. 生成模拟时间序列数据")
print("=" * 60)

np.random.seed(42)
n = 200

trend = np.linspace(20, 60, n)
seasonal = 5 * np.sin(np.arange(n) * 2 * np.pi / 12)
noise = np.random.randn(n) * 2
data = trend + seasonal + noise

dates = pd.date_range('2020-01-01', periods=n, freq='ME')
ts = pd.Series(data, index=dates, name='原始数据')

print("模拟数据 (前5个):")
print(ts.head())

# ============================================================
# 2. 简单移动平均 (Simple Moving Average, SMA)
# ============================================================
print("\n" + "=" * 60)
print("2. 简单移动平均 (SMA)")
print("=" * 60)

# rolling() 参数说明:
# - window: 窗口大小
# - min_periods: 最少需要的观测数 (默认等于 window)
# - center: 是否居中计算

# 不同窗口大小的移动平均
sma_3 = ts.rolling(window=3).mean()
sma_6 = ts.rolling(window=6).mean()
sma_12 = ts.rolling(window=12).mean()

print("3期移动平均 (前8个):")
print(sma_3.head(8))

print("\n6期移动平均 (前8个):")
print(sma_6.head(8))

print("\n12期移动平均 (前8个):")
print(sma_12.head(8))

# 居中移动平均 vs 非居中
sma_centered = ts.rolling(window=12, center=True).mean()
sma_trailing = ts.rolling(window=12, center=False).mean()
print("\n居中移动平均 vs 非居中移动平均 (第10-15行):")
comparison = pd.DataFrame({
    '原始': ts,
    '居中SMA(12)': sma_centered,
    '非居中SMA(12)': sma_trailing
})
print(comparison.iloc[10:16])

# ============================================================
# 3. 加权移动平均 (Weighted Moving Average, WMA)
# ============================================================
print("\n" + "=" * 60)
print("3. 加权移动平均 (WMA)")
print("=" * 60)

# 自定义权重: 近期数据权重更大
weights = np.array([0.1, 0.2, 0.3, 0.4])

def weighted_mean(x):
    return np.sum(x * weights)

wma = ts.rolling(window=4).apply(weighted_mean, raw=True)

print("4期加权移动平均 (前8个, 权重=[0.1,0.2,0.3,0.4]):")
print(wma.head(8))

# ============================================================
# 4. 指数加权移动平均 (Exponentially Weighted Moving Average, EWMA)
# ============================================================
print("\n" + "=" * 60)
print("4. 指数加权移动平均 (EWMA)")
print("=" * 60)

# ewm() 参数说明:
# - span: 跨度 (对应窗口大小的概念), alpha = 2/(span+1)
# - alpha: 平滑系数, 0 < alpha <= 1, 越大越重视近期数据
# - halflife: 半衰期
# - com: 质心 (center of mass)

# 不同 span 的 EWMA
ewma_span5 = ts.ewm(span=5).mean()
ewma_span12 = ts.ewm(span=12).mean()
ewma_span20 = ts.ewm(span=20).mean()

print("EWMA span=5 (前5个):")
print(ewma_span5.head())

print("\nEWMA span=12 (前5个):")
print(ewma_span12.head())

# 不同 alpha 的 EWMA
ewma_alpha_01 = ts.ewm(alpha=0.1).mean()
ewma_alpha_03 = ts.ewm(alpha=0.3).mean()
ewma_alpha_05 = ts.ewm(alpha=0.5).mean()

print("\n不同 alpha 的 EWMA 对比 (第5-10行):")
ewma_compare = pd.DataFrame({
    '原始': ts,
    'alpha=0.1': ewma_alpha_01,
    'alpha=0.3': ewma_alpha_03,
    'alpha=0.5': ewma_alpha_05
})
print(ewma_compare.iloc[5:11])

# ============================================================
# 5. 简单指数平滑 (Simple Exponential Smoothing, SES)
# ============================================================
print("\n" + "=" * 60)
print("5. 简单指数平滑 (SES) — statsmodels")
print("=" * 60)

# SimpleExpSmoothing 适用于没有明显趋势和季节性的序列
# 先对去趋势数据做 SES 演示
detrended = ts - ts.rolling(window=12).mean()
detrended_clean = detrended.dropna()

# 拟合 SES 模型
ses_model = SimpleExpSmoothing(detrended_clean, initialization_method='estimated')
ses_fit = ses_model.fit()

print(f"最优平滑系数 (alpha): {ses_fit.params['smoothing_level']:.4f}")
print(f"SSE (误差平方和): {ses_fit.sse:.4f}")
print(f"AIC: {ses_fit.aic:.4f}")

# 预测未来12期
ses_forecast = ses_fit.forecast(12)
print("\nSES 预测未来12期:")
print(ses_forecast)

# ============================================================
# 6. 不同平滑系数的 SES 对比
# ============================================================
print("\n" + "=" * 60)
print("6. 不同平滑系数的 SES 对比")
print("=" * 60)

for alpha_val in [0.1, 0.3, 0.5, 0.8]:
    ses_m = SimpleExpSmoothing(detrended_clean, initialization_method='estimated')
    ses_f = ses_m.fit(smoothing_level=alpha_val)
    print(f"alpha={alpha_val}: SSE={ses_f.sse:.2f}, AIC={ses_f.aic:.2f}")

# ============================================================
# 7. 可视化: 不同窗口的移动平均
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SMA 不同窗口
axes[0, 0].plot(ts, label='原始', alpha=0.5, linewidth=1)
axes[0, 0].plot(sma_3, label='SMA(3)', linewidth=1.5)
axes[0, 0].plot(sma_6, label='SMA(6)', linewidth=1.5)
axes[0, 0].plot(sma_12, label='SMA(12)', linewidth=2)
axes[0, 0].set_title('简单移动平均 — 不同窗口大小')
axes[0, 0].legend()

# 居中 vs 非居中
axes[0, 1].plot(ts, label='原始', alpha=0.5, linewidth=1)
axes[0, 1].plot(sma_centered, label='居中SMA(12)', linewidth=1.5)
axes[0, 1].plot(sma_trailing, label='非居中SMA(12)', linewidth=1.5)
axes[0, 1].set_title('居中 vs 非居中移动平均')
axes[0, 1].legend()

# EWMA 不同 span
axes[1, 0].plot(ts, label='原始', alpha=0.5, linewidth=1)
axes[1, 0].plot(ewma_span5, label='EWMA(span=5)', linewidth=1.5)
axes[1, 0].plot(ewma_span12, label='EWMA(span=12)', linewidth=1.5)
axes[1, 0].plot(ewma_span20, label='EWMA(span=20)', linewidth=2)
axes[1, 0].set_title('指数加权移动平均 — 不同 span')
axes[1, 0].legend()

# EWMA 不同 alpha
axes[1, 1].plot(ts, label='原始', alpha=0.5, linewidth=1)
axes[1, 1].plot(ewma_alpha_01, label='alpha=0.1', linewidth=1.5)
axes[1, 1].plot(ewma_alpha_03, label='alpha=0.3', linewidth=1.5)
axes[1, 1].plot(ewma_alpha_05, label='alpha=0.5', linewidth=2)
axes[1, 1].set_title('指数加权移动平均 — 不同 alpha')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/3_移动平均与指数平滑.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. 可视化: SES 拟合与预测
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(detrended_clean, label='去趋势数据', alpha=0.7)
ax.plot(ses_fit.fittedvalues, label=f'SES拟合 (alpha={ses_fit.params["smoothing_level"]:.3f})', linewidth=2)

forecast_index = pd.date_range(
    start=detrended_clean.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq='ME'
)
ax.plot(forecast_index, ses_forecast, label='SES预测', color='red', linewidth=2)
ax.set_title('简单指数平滑 (SES) 拟合与预测')
ax.legend()
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/3_SES拟合预测.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. 方法对比总结
# ============================================================
print("\n" + "=" * 60)
print("9. 方法对比总结")
print("=" * 60)
print("""
方法对比:
┌──────────────────┬────────────────────────────────────────────┐
│ 方法             │ 特点                                       │
├──────────────────┼────────────────────────────────────────────┤
│ 简单移动平均 SMA │ 等权重, 滞后明显, 适合平滑去噪             │
│ 加权移动平均 WMA │ 近期权重更大, 滞后减轻                     │
│ 指数加权 EWMA    │ 指数衰减权重, 只需一个参数, 计算高效       │
│ 简单指数平滑 SES │ 统计模型, 可优化参数, 可预测, 无趋势/季节  │
└──────────────────┴────────────────────────────────────────────┘

选择建议:
  - 仅需平滑: 使用 SMA 或 EWMA
  - 需要预测且无趋势/季节: 使用 SES
  - 有趋势: 使用 Holt 线性趋势法
  - 有趋势和季节: 使用 Holt-Winters 法
""")

print("图表已保存。")
