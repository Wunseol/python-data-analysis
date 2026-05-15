# -*- coding: utf-8 -*-
# 数据来源: yfinance 在线数据 (网络不可用时回退到自构建模拟数据)
# 本脚本演示完整的股票数据时间序列分析工作流

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据获取
# ============================================================
print("=" * 60)
print("1. 数据获取")
print("=" * 60)

USE_SELF_BUILT = False

try:
    import yfinance as yf
    print("正在通过 yfinance 下载股票数据...")

    ticker = 'AAPL'
    stock_data = yf.download(
        tickers=ticker,
        start='2022-01-01',
        end='2024-12-31',
        auto_adjust=True,
        progress=False
    )

    if stock_data.empty:
        raise ValueError("下载数据为空")

    print(f"成功下载 {ticker} 股票数据!")
    print(f"数据形状: {stock_data.shape}")
    USE_SELF_BUILT = False

except Exception as e:
    print(f"yfinance 下载失败: {e}")
    print("回退到自构建模拟数据...")

    np.random.seed(42)
    n_days = 750
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

    base_price = 150
    daily_returns = np.random.randn(n_days) * 0.02 + 0.0003
    close_prices = base_price * np.cumprod(1 + daily_returns)

    high_prices = close_prices * (1 + np.abs(np.random.randn(n_days) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_days) * 0.01))
    open_prices = close_prices * (1 + np.random.randn(n_days) * 0.005)
    volumes = np.random.randint(30000000, 120000000, n_days).astype(float)

    stock_data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)

    stock_data.index.name = 'Date'
    ticker = 'SIMULATED'
    USE_SELF_BUILT = True
    print(f"模拟数据生成完成, 形状: {stock_data.shape}")

# ============================================================
# 2. 数据概览
# ============================================================
print("\n" + "=" * 60)
print("2. 数据概览")
print("=" * 60)

print("前5行:")
print(stock_data.head())
print(f"\n数据信息:")
print(stock_data.dtypes)
print(f"\n基本统计:")
print(stock_data.describe())

# 确保索引为 DatetimeIndex
if not isinstance(stock_data.index, pd.DatetimeIndex):
    stock_data.index = pd.to_datetime(stock_data.index)

# 提取收盘价作为主要分析对象
if isinstance(stock_data.columns, pd.MultiIndex):
    close = stock_data[('Close', ticker)].copy()
    close.name = 'Close'
else:
    close = stock_data['Close'].copy()

print(f"\n收盘价统计:")
print(f"  均值: {close.mean():.2f}")
print(f"  标准差: {close.std():.2f}")
print(f"  最小值: {close.min():.2f}")
print(f"  最大值: {close.max():.2f}")

# ============================================================
# 3. 数据可视化
# ============================================================
print("\n" + "=" * 60)
print("3. 数据可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 收盘价走势
axes[0].plot(close, color='steelblue', linewidth=1)
axes[0].set_title(f'{ticker} 收盘价走势')
axes[0].set_ylabel('价格')
axes[0].grid(True, alpha=0.3)

# 成交量
if isinstance(stock_data.columns, pd.MultiIndex):
    volume = stock_data[('Volume', ticker)]
else:
    volume = stock_data['Volume']

axes[1].bar(volume.index, volume.values, color='gray', alpha=0.5, width=2)
axes[1].set_title(f'{ticker} 成交量')
axes[1].set_ylabel('成交量')
axes[1].set_xlabel('日期')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_价格走势.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. 时间序列分解
# ============================================================
print("\n" + "=" * 60)
print("4. 时间序列分解")
print("=" * 60)

# 使用月度数据做分解 (日数据噪声太大)
close_monthly = close.resample('ME').mean()
print(f"月度收盘价数据量: {len(close_monthly)}")

decomposition = seasonal_decompose(close_monthly, model='multiplicative', period=12)

fig = decomposition.plot()
fig.set_size_inches(14, 10)
fig.suptitle(f'{ticker} 收盘价时间序列分解 (乘法模型)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_分解.png', dpi=150, bbox_inches='tight')
plt.show()

print("分解结果:")
print(f"  趋势成分范围: {decomposition.trend.min():.2f} ~ {decomposition.trend.max():.2f}")
print(f"  季节成分范围: {decomposition.seasonal.min():.4f} ~ {decomposition.seasonal.max():.4f}")

# ============================================================
# 5. 移动平均分析
# ============================================================
print("\n" + "=" * 60)
print("5. 移动平均分析")
print("=" * 60)

ma_20 = close.rolling(window=20).mean()
ma_50 = close.rolling(window=50).mean()
ma_200 = close.rolling(window=200).mean()

print(f"20日均线 (最新): {ma_20.iloc[-1]:.2f}")
print(f"50日均线 (最新): {ma_50.iloc[-1]:.2f}")
print(f"200日均线 (最新): {ma_200.iloc[-1]:.2f}")

# 均线交叉信号
if ma_20.iloc[-1] > ma_50.iloc[-1]:
    print("当前信号: 20日均线 > 50日均线 → 看多 (金叉)")
else:
    print("当前信号: 20日均线 < 50日均线 → 看空 (死叉)")

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(close, label='收盘价', alpha=0.6, linewidth=1)
ax.plot(ma_20, label='20日均线', linewidth=1.5)
ax.plot(ma_50, label='50日均线', linewidth=1.5)
ax.plot(ma_200, label='200日均线', linewidth=2, color='red')
ax.set_title(f'{ticker} 收盘价与移动平均线')
ax.set_ylabel('价格')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_移动平均.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6. 平稳性检验
# ============================================================
print("\n" + "=" * 60)
print("6. 平稳性检验")
print("=" * 60)

def adf_test(series, title=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF 检验: {title}")
    print(f"  统计量: {result[0]:.4f}, p值: {result[1]:.6f}")
    if result[1] < 0.05:
        print(f"  结论: 平稳 (p < 0.05)")
    else:
        print(f"  结论: 非平稳 (p >= 0.05)")
    return result[1] < 0.05

adf_test(close, '原始收盘价')

close_diff = close.diff().dropna()
is_stationary = adf_test(close_diff, '一阶差分 (日收益率)')

# ============================================================
# 7. ACF/PACF 分析
# ============================================================
print("\n" + "=" * 60)
print("7. ACF/PACF 分析")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_acf(close_diff, lags=40, ax=axes[0, 0], title='日收益率 ACF')
plot_pacf(close_diff, lags=40, ax=axes[0, 1], title='日收益率 PACF')

# 对原始序列的 ACF/PACF
plot_acf(close, lags=40, ax=axes[1, 0], title='原始收盘价 ACF')
plot_pacf(close, lags=40, ax=axes[1, 1], title='原始收盘价 PACF')

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_ACF_PACF.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. ARIMA 建模
# ============================================================
print("\n" + "=" * 60)
print("8. ARIMA 建模")
print("=" * 60)

# 使用日收益率建模 (已平稳)
# 网格搜索最优参数
best_aic = np.inf
best_order = None

print("网格搜索 ARIMA 参数 (p,0,q)...")
for p in range(0, 4):
    for q in range(0, 4):
        try:
            model_temp = ARIMA(close_diff, order=(p, 0, q))
            fit_temp = model_temp.fit()
            if fit_temp.aic < best_aic:
                best_aic = fit_temp.aic
                best_order = (p, 0, q)
        except Exception:
            continue

print(f"最优参数: ARIMA{best_order}, AIC={best_aic:.2f}")

# 也可以直接对原始序列拟合 ARIMA(p,1,q)
best_aic_raw = np.inf
best_order_raw = None

for p in range(0, 4):
    for q in range(0, 4):
        try:
            model_temp = ARIMA(close, order=(p, 1, q))
            fit_temp = model_temp.fit()
            if fit_temp.aic < best_aic_raw:
                best_aic_raw = fit_temp.aic
                best_order_raw = (p, 1, q)
        except Exception:
            continue

print(f"原始序列最优参数: ARIMA{best_order_raw}, AIC={best_aic_raw:.2f}")

# 拟合最优模型
model = ARIMA(close, order=best_order_raw)
fitted = model.fit()
print(f"\nARIMA{best_order_raw} 模型摘要:")
print(fitted.summary())

# ============================================================
# 9. 残差诊断
# ============================================================
print("\n" + "=" * 60)
print("9. 残差诊断")
print("=" * 60)

residuals = fitted.resid
print(f"残差均值: {residuals.mean():.6f}")
print(f"残差标准差: {residuals.std():.4f}")

from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"\nLjung-Box 检验:")
print(lb_test)
if lb_test['lb_pvalue'].values[0] > 0.05:
    print("残差为白噪声, 模型拟合充分 ✓")
else:
    print("残差非白噪声, 模型可能需要改进 ✗")

fig = fitted.plot_diagnostics(figsize=(14, 10))
plt.suptitle(f'ARIMA{best_order_raw} 残差诊断', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_残差诊断.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 10. 预测
# ============================================================
print("\n" + "=" * 60)
print("10. 预测")
print("=" * 60)

# 划分训练集和测试集
train_size = int(len(close) * 0.8)
train = close.iloc[:train_size]
test = close.iloc[train_size:]

print(f"训练集: {train.index[0].date()} ~ {train.index[-1].date()} ({len(train)}条)")
print(f"测试集: {test.index[0].date()} ~ {test.index[-1].date()} ({len(test)}条)")

# 用训练集重新拟合
model_train = ARIMA(train, order=best_order_raw)
fitted_train = model_train.fit()

# 预测
forecast_steps = len(test)
forecast_result = fitted_train.get_forecast(steps=forecast_steps)
forecast_values = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

forecast_values.index = test.index
conf_int.index = test.index

# 评估
mae = np.mean(np.abs(test.values - forecast_values.values))
rmse = np.sqrt(np.mean((test.values - forecast_values.values) ** 2))
mape = np.mean(np.abs((test.values - forecast_values.values) / test.values)) * 100

print(f"\n预测评估:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAPE: {mape:.2f}%")

# ============================================================
# 11. 预测可视化
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 全景图
axes[0].plot(train, label='训练集', color='steelblue', alpha=0.7)
axes[0].plot(test, label='测试集(真实)', color='green')
axes[0].plot(forecast_values, label=f'ARIMA{best_order_raw} 预测', color='red', linewidth=2)
axes[0].fill_between(
    conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
    alpha=0.2, color='red', label='95% 置信区间'
)
axes[0].set_title(f'{ticker} ARIMA{best_order_raw} 预测结果')
axes[0].set_ylabel('价格')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 放大测试集
axes[1].plot(test, label='真实值', color='green', linewidth=2)
axes[1].plot(forecast_values, label='预测值', color='red', linewidth=2)
axes[1].fill_between(
    conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
    alpha=0.2, color='red', label='95% 置信区间'
)
axes[1].set_title('预测结果放大 (测试集)')
axes[1].set_ylabel('价格')
axes[1].set_xlabel('日期')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_预测结果.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 12. 未来预测
# ============================================================
print("\n" + "=" * 60)
print("12. 未来30天预测")
print("=" * 60)

model_full = ARIMA(close, order=best_order_raw)
fitted_full = model_full.fit()

future_forecast = fitted_full.get_forecast(steps=30)
future_values = future_forecast.predicted_mean
future_conf = future_forecast.conf_int(alpha=0.05)

future_dates = pd.date_range(
    start=close.index[-1] + pd.tseries.offsets.BDay(1),
    periods=30,
    freq='B'
)
future_values.index = future_dates
future_conf.index = future_dates

print("未来30天预测 (前5个):")
for i in range(5):
    print(f"  {future_dates[i].date()}: {future_values.iloc[i]:.2f} "
          f"[{future_conf.iloc[i, 0]:.2f}, {future_conf.iloc[i, 1]:.2f}]")

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(close.iloc[-100:], label='历史收盘价', color='steelblue')
ax.plot(future_values, label='未来30天预测', color='red', linewidth=2, marker='o', markersize=3)
ax.fill_between(
    future_conf.index, future_conf.iloc[:, 0], future_conf.iloc[:, 1],
    alpha=0.2, color='red', label='95% 置信区间'
)
ax.axvline(x=close.index[-1], color='gray', linestyle='--', alpha=0.5, label='预测起点')
ax.set_title(f'{ticker} 未来30天价格预测')
ax.set_ylabel('价格')
ax.set_xlabel('日期')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/9_未来预测.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 13. 分析总结
# ============================================================
print("\n" + "=" * 60)
print("13. 分析总结")
print("=" * 60)
print(f"""
股票时间序列分析总结 ({ticker}):
{'=' * 50}
1. 数据特征:
   - 数据来源: {'yfinance 在线数据' if not USE_SELF_BUILT else '自构建模拟数据'}
   - 数据量: {len(close)} 个交易日
   - 价格范围: {close.min():.2f} ~ {close.max():.2f}

2. 平稳性:
   - 原始序列: 非平稳 (存在趋势)
   - 一阶差分: {'平稳' if is_stationary else '需进一步检验'}

3. 模型:
   - 最优 ARIMA 参数: {best_order_raw}
   - 测试集 MAPE: {mape:.2f}%

4. 注意事项:
   - 股票价格预测具有高度不确定性
   - ARIMA 模型的置信区间会随预测步长增大而变宽
   - 实际投资决策不应仅依赖统计模型
   - 建议结合基本面分析和技术分析
""")

print("所有图表已保存。")
