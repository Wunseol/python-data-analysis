# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据
# 本脚本演示时间序列预测与评估方法

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成模拟数据
# ============================================================
print("=" * 60)
print("1. 生成模拟数据")
print("=" * 60)

np.random.seed(42)
n = 300

# 生成带趋势和季节性的数据
trend = np.linspace(50, 120, n)
seasonal = 8 * np.sin(np.arange(n) * 2 * np.pi / 12)
noise = np.random.randn(n) * 3
data = trend + seasonal + noise

dates = pd.date_range('2020-01-01', periods=n, freq='D')
ts = pd.Series(data, index=dates, name='模拟数据')

print(f"数据长度: {len(ts)}")
print(f"数据范围: {ts.index[0].date()} ~ {ts.index[-1].date()}")
print(f"均值: {ts.mean():.2f}, 标准差: {ts.std():.2f}")

# ============================================================
# 2. 时间序列的训练集/测试集划分
# ============================================================
print("\n" + "=" * 60)
print("2. 时间序列的训练集/测试集划分")
print("=" * 60)

# 重要: 时间序列不能随机划分, 必须按时间顺序划分
train_size = int(len(ts) * 0.8)
train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

print(f"训练集: {train.index[0].date()} ~ {train.index[-1].date()}, 共 {len(train)} 条")
print(f"测试集: {test.index[0].date()} ~ {test.index[-1].date()}, 共 {len(test)} 条")

# ============================================================
# 3. 确定模型参数
# ============================================================
print("\n" + "=" * 60)
print("3. 确定模型参数")
print("=" * 60)

# ADF 检验确定 d
adf_result = adfuller(train, autolag='AIC')
print(f"训练集 ADF: 统计量={adf_result[0]:.4f}, p={adf_result[1]:.6f}")

if adf_result[1] >= 0.05:
    train_diff = train.diff().dropna()
    adf_diff = adfuller(train_diff, autolag='AIC')
    print(f"一阶差分 ADF: 统计量={adf_diff[0]:.4f}, p={adf_diff[1]:.6f}")
    d = 1
else:
    d = 0

print(f"选择 d = {d}")

# 网格搜索最优 (p,d,q)
best_aic = np.inf
best_order = None

for p in range(0, 4):
    for q in range(0, 4):
        try:
            model_temp = ARIMA(train, order=(p, d, q))
            fit_temp = model_temp.fit()
            if fit_temp.aic < best_aic:
                best_aic = fit_temp.aic
                best_order = (p, d, q)
        except Exception:
            continue

print(f"最优参数: ARIMA{best_order}, AIC={best_aic:.2f}")

# ============================================================
# 4. 拟合模型并进行预测
# ============================================================
print("\n" + "=" * 60)
print("4. 拟合模型并进行预测")
print("=" * 60)

model = ARIMA(train, order=best_order)
fitted = model.fit()

# forecast() 参数说明:
# - steps: 预测步数
# 返回: 预测值, 标准误差, 置信区间

forecast_steps = len(test)
forecast_result = fitted.get_forecast(steps=forecast_steps)
forecast_values = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)

print(f"预测步数: {forecast_steps}")
print(f"\n预测值 (前5个):")
print(forecast_values.head())
print(f"\n95% 置信区间 (前5个):")
print(conf_int.head())

# ============================================================
# 5. 预测评估指标
# ============================================================
print("\n" + "=" * 60)
print("5. 预测评估指标")
print("=" * 60)

# 对齐索引
forecast_values.index = test.index
conf_int.index = test.index

# MAE (Mean Absolute Error, 平均绝对误差)
mae = np.mean(np.abs(test.values - forecast_values.values))
print(f"MAE  (平均绝对误差): {mae:.4f}")

# RMSE (Root Mean Squared Error, 均方根误差)
rmse = np.sqrt(np.mean((test.values - forecast_values.values) ** 2))
print(f"RMSE (均方根误差):   {rmse:.4f}")

# MAPE (Mean Absolute Percentage Error, 平均绝对百分比误差)
# 注意: 当真实值接近0时 MAPE 会很大, 需谨慎使用
mape = np.mean(np.abs((test.values - forecast_values.values) / test.values)) * 100
print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")

print("""
评估指标解读:
  - MAE: 误差的绝对值平均, 与原始数据同量纲, 直观易理解
  - RMSE: 对大误差更敏感 (因为平方), 常用于模型比较
  - MAPE: 百分比误差, 便于跨数据集比较
    MAPE < 10%: 优秀预测
    MAPE 10%-20%: 良好预测
    MAPE 20%-50%: 合理预测
    MAPE > 50%: 预测不准确
""")

# ============================================================
# 6. 封装评估函数
# ============================================================
print("=" * 60)
print("6. 封装评估函数")
print("=" * 60)

def evaluate_forecast(actual, predicted):
    """计算并返回常用预测评估指标"""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # R² (决定系数)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

print("ARIMA 预测评估:")
metrics = evaluate_forecast(test.values, forecast_values.values)

# ============================================================
# 7. 不同预测方法的对比
# ============================================================
print("\n" + "=" * 60)
print("7. 不同预测方法的对比")
print("=" * 60)

# 方法1: 朴素法 (Naive) — 用最后一个训练值作为预测
naive_forecast = pd.Series(
    np.full(forecast_steps, train.iloc[-1]),
    index=test.index
)
print("朴素法 (Naive):")
metrics_naive = evaluate_forecast(test.values, naive_forecast.values)

# 方法2: 季节朴素法 — 用一年前的值作为预测
seasonal_naive_forecast = pd.Series(
    train.iloc[-forecast_steps:].values,
    index=test.index
)
print("\n季节朴素法:")
metrics_seasonal = evaluate_forecast(test.values, seasonal_naive_forecast.values)

# 方法3: 移动平均法
ma_forecast = pd.Series(
    np.full(forecast_steps, train.iloc[-12:].mean()),
    index=test.index
)
print("\n移动平均法 (窗口=12):")
metrics_ma = evaluate_forecast(test.values, ma_forecast.values)

# 方法4: ARIMA
print(f"\nARIMA{best_order}:")
metrics_arima = evaluate_forecast(test.values, forecast_values.values)

# ============================================================
# 8. 可视化: 预测结果
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 主图: ARIMA 预测与置信区间
axes[0].plot(train, label='训练集', color='steelblue')
axes[0].plot(test, label='测试集(真实)', color='green')
axes[0].plot(forecast_values, label=f'ARIMA{best_order} 预测', color='red', linewidth=2)
axes[0].fill_between(
    conf_int.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha=0.2, color='red', label='95% 置信区间'
)
axes[0].set_title(f'ARIMA{best_order} 预测结果')
axes[0].legend()

# 放大测试集区域
axes[1].plot(test, label='测试集(真实)', color='green', linewidth=2)
axes[1].plot(forecast_values, label='ARIMA 预测', color='red', linewidth=2)
axes[1].fill_between(
    conf_int.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha=0.2, color='red', label='95% 置信区间'
)
axes[1].set_title('预测结果放大 (测试集)')
axes[1].legend()

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/7_ARIMA预测.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. 可视化: 方法对比
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test, label='真实值', color='green', linewidth=2)
ax.plot(forecast_values, label=f'ARIMA{best_order}', color='red', linewidth=1.5)
ax.plot(naive_forecast, label='朴素法', color='gray', linestyle='--')
ax.plot(seasonal_naive_forecast, label='季节朴素法', color='purple', linestyle='--')
ax.plot(ma_forecast, label='移动平均法', color='orange', linestyle='--')
ax.set_title('不同预测方法对比')
ax.legend()
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/7_方法对比.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 10. 可视化: 误差分析
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

errors = test.values - forecast_values.values

# 误差分布
axes[0].hist(errors, bins=20, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_title('预测误差分布')
axes[0].set_xlabel('误差')
axes[0].set_ylabel('频数')

# 误差时间序列
axes[1].plot(test.index, errors, marker='o', markersize=3, linewidth=1)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_title('预测误差随时间变化')
axes[1].set_xlabel('日期')
axes[1].set_ylabel('误差')

plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/7_误差分析.png', dpi=150, bbox_inches='tight')
plt.show()

print("图表已保存。")
