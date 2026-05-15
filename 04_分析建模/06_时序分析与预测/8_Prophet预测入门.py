# -*- coding: utf-8 -*-
# 数据来源: 自构建模拟数据
# [注意] fbprophet 已更名为 prophet，请使用 pip install prophet 安装
# 本脚本演示 Prophet 预测库的基本用法

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Prophet 简介
# ============================================================
print("=" * 60)
print("1. Prophet 简介")
print("=" * 60)
print("""
Prophet 是 Meta (Facebook) 开源的时间序列预测库, 特点:
  - 自动检测趋势变化点
  - 自动处理季节性 (年/周/日)
  - 支持节假日效应
  - 对缺失值和异常值鲁棒
  - 无需深入的时间序列知识即可使用

安装: pip install prophet
  [注意] fbprophet 已更名为 prophet，请使用 pip install prophet 安装

Prophet 要求输入数据格式:
  - ds: 日期列 (datetime 类型)
  - y: 数值列 (float 类型)
""")

# ============================================================
# 2. 构建模拟数据
# ============================================================
print("=" * 60)
print("2. 构建模拟数据")
print("=" * 60)

np.random.seed(42)
n_days = 730

dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

# 趋势: 分段线性增长
trend = np.piecewise(
    np.arange(n_days, dtype=float),
    [np.arange(n_days) < 365, np.arange(n_days) >= 365],
    [lambda x: 100 + 0.1 * x, lambda x: 136.5 + 0.2 * (x - 365)]
)

# 年季节性
yearly = 15 * np.sin(np.arange(n_days) * 2 * np.pi / 365.25)

# 周季节性
weekly = 3 * np.sin(np.arange(n_days) * 2 * np.pi / 7)

# 噪声
noise = np.random.randn(n_days) * 5

# 合成
y = trend + yearly + weekly + noise

df = pd.DataFrame({'ds': dates, 'y': y})

print("Prophet 输入数据 (前5行):")
print(df.head())
print(f"\n数据形状: {df.shape}")
print(f"日期范围: {df['ds'].min().date()} ~ {df['ds'].max().date()}")

# ============================================================
# 3. 创建并拟合 Prophet 模型
# ============================================================
print("\n" + "=" * 60)
print("3. 创建并拟合 Prophet 模型")
print("=" * 60)

from prophet import Prophet

# 创建 Prophet 模型
# 常用参数:
#   growth: 'linear'(线性趋势) 或 'logistic'(逻辑增长)
#   seasonality_mode: 'additive'(加法) 或 'multiplicative'(乘法)
#   yearly_seasonality: 是否自动检测年季节性 (True/False/整数=傅里叶阶数)
#   weekly_seasonality: 是否自动检测周季节性
#   daily_seasonality: 是否自动检测日季节性
#   changepoint_prior_scale: 趋势变化点灵活度 (默认0.05, 越大越灵活)
#   seasonality_prior_scale: 季节性强度 (默认10)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05
)

# 拟合模型
model.fit(df)
print("模型拟合完成!")

# ============================================================
# 4. 创建未来日期数据框
# ============================================================
print("\n" + "=" * 60)
print("4. 创建未来日期数据框")
print("=" * 60)

# make_future_dataframe 参数:
# - periods: 预测未来的天数
# - freq: 频率 ('D'=天, 'W'=周, 'ME'=月)
# - include_history: 是否包含历史数据

future = model.make_future_dataframe(periods=90, freq='D', include_history=True)
print(f"未来数据框形状: {future.shape}")
print(f"日期范围: {future['ds'].min().date()} ~ {future['ds'].max().date()}")
print("\n未来数据框 (最后5行):")
print(future.tail())

# ============================================================
# 5. 进行预测
# ============================================================
print("\n" + "=" * 60)
print("5. 进行预测")
print("=" * 60)

forecast = model.predict(future)

# 预测结果包含的列
print("预测结果列名:")
print(forecast.columns.tolist())

# 关键列说明:
# - ds: 日期
# - yhat: 预测值
# - yhat_lower: 预测下界
# - yhat_upper: 预测上界
# - trend: 趋势成分
# - yearly: 年季节性成分
# - weekly: 周季节性成分

print("\n预测结果 (最后5行, 关键列):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].tail())

# 仅查看未来预测部分
future_only = forecast[forecast['ds'] > df['ds'].max()]
print(f"\n未来90天预测 (前5行):")
print(future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# ============================================================
# 6. 绘制预测图
# ============================================================
print("\n" + "=" * 60)
print("6. 绘制预测图")
print("=" * 60)

fig1 = model.plot(forecast, figsize=(14, 6))
ax = fig1.axes[0]
ax.set_title('Prophet 预测结果')
ax.set_xlabel('日期')
ax.set_ylabel('值')
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/8_Prophet预测.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. 绘制成分分解图
# ============================================================
print("=" * 60)
print("7. 绘制成分分解图")
print("=" * 60)

fig2 = model.plot_components(forecast, figsize=(14, 10))
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/8_Prophet成分分解.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
成分分解图说明:
  - trend: 趋势成分, 显示长期变化方向
  - yearly: 年季节性, 显示一年内的周期性模式
  - weekly: 周季节性, 显示一周内的周期性模式
""")

# ============================================================
# 8. 添加自定义季节性
# ============================================================
print("=" * 60)
print("8. 添加自定义季节性")
print("=" * 60)

model_custom = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# 添加月季节性 (period=30.5, fourier_order=5)
model_custom.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

model_custom.fit(df)
future_custom = model_custom.make_future_dataframe(periods=90)
forecast_custom = model_custom.predict(future_custom)

fig3 = model_custom.plot_components(forecast_custom, figsize=(14, 12))
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/8_自定义季节性.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. 调整趋势灵活度
# ============================================================
print("=" * 60)
print("9. 调整趋势灵活度 (changepoint_prior_scale)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, cps in enumerate([0.01, 0.05, 0.5]):
    m = Prophet(
        changepoint_prior_scale=cps,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    m.fit(df)
    fcst = m.predict(m.make_future_dataframe(periods=90))

    axes[idx].plot(df['ds'], df['y'], '.', markersize=2, alpha=0.5, label='实际')
    axes[idx].plot(fcst['ds'], fcst['yhat'], 'r-', linewidth=1, label='预测')
    axes[idx].fill_between(
        fcst['ds'], fcst['yhat_lower'], fcst['yhat_upper'],
        alpha=0.2, color='red'
    )
    axes[idx].set_title(f'changepoint_prior_scale={cps}')
    axes[idx].legend()

plt.suptitle('趋势灵活度对比', fontsize=14)
plt.tight_layout()
plt.savefig('d:/Dev/DevWorkSpace/VS Code/Python/python-data-analysis/3_2_时间序列分析/8_趋势灵活度.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
changepoint_prior_scale 参数说明:
  - 值越小 (如0.01): 趋势越平滑, 欠拟合风险
  - 默认值 (0.05): 适中
  - 值越大 (如0.5): 趋势越灵活, 过拟合风险
""")

# ============================================================
# 10. 交叉验证 (可选)
# ============================================================
print("=" * 60)
print("10. 交叉验证概念")
print("=" * 60)
print("""
Prophet 提供了时间序列交叉验证功能:

  from prophet.diagnostics import cross_validation, performance_metrics

  # 初始训练期: 365天, 每隔90天做一次预测, 每次预测30天
  df_cv = cross_validation(model, initial='365 days', period='90 days', horizon='30 days')

  # 计算评估指标
  df_p = performance_metrics(df_cv)
  print(df_p.head())

  评估指标包括: MSE, RMSE, MAE, MAPE, MDAPE, SMAPE, Coverage
""")

# ============================================================
# 11. Prophet vs ARIMA 对比总结
# ============================================================
print("=" * 60)
print("11. Prophet vs ARIMA 对比总结")
print("=" * 60)
print("""
┌──────────────┬──────────────────────────┬──────────────────────────┐
│ 特性         │ Prophet                  │ ARIMA                    │
├──────────────┼──────────────────────────┼──────────────────────────┤
│ 使用难度     │ 低, 自动化程度高         │ 较高, 需手动选参         │
│ 趋势处理     │ 自动检测变化点           │ 需差分处理               │
│ 季节性       │ 自动处理多周期季节性     │ 需SARIMA扩展             │
│ 节假日       │ 内置支持                 │ 不直接支持               │
│ 缺失值       │ 鲁棒                    │ 需预处理                 │
│ 可解释性     │ 成分分解直观             │ 需专业知识解读           │
│ 适用场景     │ 商业预测, 日数据         │ 统计建模, 学术研究       │
└──────────────┴──────────────────────────┴──────────────────────────┘
""")

print("图表已保存。")
