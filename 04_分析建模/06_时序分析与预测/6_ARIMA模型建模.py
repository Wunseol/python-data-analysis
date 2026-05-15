# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据 (ARIMA过程)
# 本脚本演示 ARIMA 模型的建模过程

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. ARIMA 模型概念
# ============================================================
print("=" * 60)
print("1. ARIMA 模型概念")
print("=" * 60)
print("""
ARIMA(p, d, q) 模型:
  AR (AutoRegressive, 自回归): 用过去的值预测当前值
    y_t = c + φ1*y_{t-1} + φ2*y_{t-2} + ... + φp*y_{t-p} + ε_t

  I (Integrated, 差分): 通过 d 阶差分使序列平稳
    Δ^d y_t

  MA (Moving Average, 滑动平均): 用过去的误差预测当前值
    y_t = c + ε_t + θ1*ε_{t-1} + θ2*ε_{t-2} + ... + θq*ε_{t-q}

  参数含义:
    p: 自回归阶数 (AR项)
    d: 差分阶数 (使序列平稳所需的差分次数)
    q: 滑动平均阶数 (MA项)

  特殊情况:
    ARIMA(p,0,0) = AR(p)
    ARIMA(0,0,q) = MA(q)
    ARIMA(0,1,0) = 随机游走
    ARIMA(p,0,q) = ARMA(p,q)
""")

# ============================================================
# 2. 生成模拟数据
# ============================================================
print("=" * 60)
print("2. 生成模拟数据")
print("=" * 60)

np.random.seed(42)
n = 300

# 生成 ARIMA(1,1,1) 过程的数据
# 先生成 ARMA(1,1) 的平稳数据, 再做累积和得到非平稳序列
ar_coef = 0.6
ma_coef = 0.4
errors = np.random.randn(n)

arma_data = np.zeros(n)
for i in range(1, n):
    arma_data[i] = ar_coef * arma_data[i - 1] + errors[i] + ma_coef * errors[i - 1]

# 累积和使其变为 I(1) 过程
arima_data = np.cumsum(arma_data) + 100

dates = pd.date_range('2020-01-01', periods=n, freq='D')
ts = pd.Series(arima_data, index=dates, name='模拟ARIMA数据')

print("模拟数据 (前5个):")
print(ts.head())
print(f"\n数据长度: {len(ts)}")

# ============================================================
# 3. 确定差分阶数 d
# ============================================================
print("\n" + "=" * 60)
print("3. 确定差分阶数 d")
print("=" * 60)

# 原始序列 ADF 检验
adf_orig = adfuller(ts, autolag='AIC')
print(f"原始序列 ADF: 统计量={adf_orig[0]:.4f}, p={adf_orig[1]:.6f}")
if adf_orig[1] < 0.05:
    print("  → 原始序列平稳, d=0")
else:
    print("  → 原始序列非平稳, 需要差分")

# 一阶差分 ADF 检验
ts_diff1 = ts.diff().dropna()
adf_diff1 = adfuller(ts_diff1, autolag='AIC')
print(f"\n一阶差分 ADF: 统计量={adf_diff1[0]:.4f}, p={adf_diff1[1]:.6f}")
if adf_diff1[1] < 0.05:
    print("  → 一阶差分后平稳, d=1")
else:
    print("  → 一阶差分后仍非平稳, 需要二阶差分")

# 二阶差分 ADF 检验
ts_diff2 = ts_diff1.diff().dropna()
adf_diff2 = adfuller(ts_diff2, autolag='AIC')
print(f"\n二阶差分 ADF: 统计量={adf_diff2[0]:.4f}, p={adf_diff2[1]:.6f}")

d = 1
print(f"\n选择 d = {d}")

# ============================================================
# 4. 确定 p 和 q — 通过 ACF/PACF 图
# ============================================================
print("\n" + "=" * 60)
print("4. 确定 p 和 q — 通过 ACF/PACF 图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts_diff1, lags=30, ax=axes[0], title='一阶差分 — ACF')
plot_pacf(ts_diff1, lags=30, ax=axes[1], title='一阶差分 — PACF')
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/6_差分ACF_PACF.png', dpi=150, bbox_inches='tight')
plt.show()

# 通过 ACF/PACF 数值辅助判断
acf_vals = acf(ts_diff1, nlags=10)
pacf_vals = pacf(ts_diff1, nlags=10)

print("一阶差分的 ACF (滞后1-10):")
for i in range(1, 11):
    print(f"  滞后{i:2d}: ACF={acf_vals[i]:+.4f}, PACF={pacf_vals[i]:+.4f}")

print("""
根据 ACF/PACF 图初步判断:
  - ACF 在滞后1后截尾 → q=1
  - PACF 在滞后1后截尾 → p=1
  → 初步选择 ARIMA(1,1,1)
""")

# ============================================================
# 5. 拟合 ARIMA 模型
# ============================================================
print("=" * 60)
print("5. 拟合 ARIMA 模型")
print("=" * 60)

# ARIMA 参数说明:
# - endog: 时间序列数据
# - order: (p, d, q) 元组

model = ARIMA(ts, order=(1, 1, 1))
fitted_model = model.fit()

print("ARIMA(1,1,1) 拟合结果:")
print(fitted_model.summary())

# ============================================================
# 6. 模型摘要解读
# ============================================================
print("\n" + "=" * 60)
print("6. 模型摘要解读")
print("=" * 60)
print(f"""
模型摘要关键信息:
  - ar.L1: 自回归系数 φ1 = {fitted_model.params[0]:.4f}
    P>|z| = {fitted_model.pvalues[0]:.4f} {'(显著)' if fitted_model.pvalues[0] < 0.05 else '(不显著)'}

  - ma.L1: 滑动平均系数 θ1 = {fitted_model.params[1]:.4f}
    P>|z| = {fitted_model.pvalues[1]:.4f} {'(显著)' if fitted_model.pvalues[1] < 0.05 else '(不显著)'}

  - sigma2: 误差方差 = {fitted_model.params[2]:.4f}

  - AIC: {fitted_model.aic:.2f} (越小越好)
  - BIC: {fitted_model.bic:.2f} (越小越好)
  - Log Likelihood: {fitted_model.llf:.2f}
""")

# ============================================================
# 7. 尝试不同 (p,d,q) 组合 — 网格搜索
# ============================================================
print("=" * 60)
print("7. 网格搜索最优 (p,d,q)")
print("=" * 60)

best_aic = np.inf
best_order = None
results = []

for p in range(0, 4):
    for q in range(0, 4):
        try:
            model_temp = ARIMA(ts, order=(p, d, q))
            fit_temp = model_temp.fit()
            results.append((p, d, q, fit_temp.aic, fit_temp.bic))
            if fit_temp.aic < best_aic:
                best_aic = fit_temp.aic
                best_order = (p, d, q)
        except Exception:
            continue

print("各 (p,d,q) 组合的 AIC/BIC:")
print(f"{'(p,d,q)':<12} {'AIC':>10} {'BIC':>10}")
print("-" * 35)
for p, d_val, q, aic, bic in sorted(results, key=lambda x: x[3]):
    marker = " ← 最优" if (p, d_val, q) == best_order else ""
    print(f"({p},{d_val},{q}){'':<5} {aic:>10.2f} {bic:>10.2f}{marker}")

print(f"\n最优参数: ARIMA{best_order}, AIC={best_aic:.2f}")

# ============================================================
# 8. 使用最优参数拟合模型
# ============================================================
print("\n" + "=" * 60)
print("8. 使用最优参数拟合模型")
print("=" * 60)

best_model = ARIMA(ts, order=best_order)
best_fit = best_model.fit()
print(f"ARIMA{best_order} 拟合结果摘要:")
print(best_fit.summary())

# ============================================================
# 9. 残差分析
# ============================================================
print("\n" + "=" * 60)
print("9. 残差分析")
print("=" * 60)

residuals = best_fit.resid

print("残差统计:")
print(f"  均值: {residuals.mean():.6f}")
print(f"  标准差: {residuals.std():.4f}")
print(f"  偏度: {residuals.skew():.4f}")
print(f"  峰度: {residuals.kurtosis():.4f}")

# 残差的 ADF 检验
adf_resid = adfuller(residuals, autolag='AIC')
print(f"\n残差 ADF 检验: 统计量={adf_resid[0]:.4f}, p={adf_resid[1]:.6f}")
if adf_resid[1] < 0.05:
    print("  → 残差平稳 ✓")
else:
    print("  → 残差非平稳 ✗")

# Ljung-Box 检验 (检验残差是否为白噪声)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"\nLjung-Box 检验 (滞后10):")
print(lb_test)
if lb_test['lb_pvalue'].values[0] > 0.05:
    print("  → 残差为白噪声, 模型拟合充分 ✓")
else:
    print("  → 残差非白噪声, 模型可能需要改进 ✗")

# ============================================================
# 10. 残差诊断图
# ============================================================
fig = best_fit.plot_diagnostics(figsize=(14, 10))
plt.suptitle(f'ARIMA{best_order} 残差诊断', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/6_残差诊断.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 11. 拟合值与原始数据对比
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts, label='原始数据', alpha=0.7)
ax.plot(best_fit.fittedvalues, label=f'ARIMA{best_order} 拟合值', alpha=0.8, linewidth=1.5)
ax.set_title(f'ARIMA{best_order} 拟合效果')
ax.legend()
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/6_拟合效果.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存。")
