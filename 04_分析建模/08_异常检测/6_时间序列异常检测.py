# 数据来源: 模拟时间序列数据

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)

# 生成模拟时间序列
n_days = 365
dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")
trend = np.linspace(50, 60, n_days)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
noise = np.random.normal(0, 2, n_days)
values = trend + seasonality + noise

# 注入异常
anomaly_indices = [30, 80, 150, 200, 280, 320]
for idx in anomaly_indices:
    values[idx] += np.random.choice([-1, 1]) * np.random.uniform(15, 25)

y_true = np.zeros(n_days, dtype=int)
y_true[anomaly_indices] = 1

ts = pd.Series(values, index=dates, name="值")
print(f"时间序列长度: {len(ts)}")
print(f"异常点数: {y_true.sum()}")
print(f"异常日期: {ts.index[y_true == 1].strftime('%Y-%m-%d').tolist()}")

# === 移动平均+残差法 ===
print("\n=== 移动平均+残差法 ===")

window = 7
ma = ts.rolling(window=window, center=True).mean()
residual = ts - ma
residual_std = residual.std()

# 3σ残差异常
ma_anomaly = (np.abs(residual) > 3 * residual_std).astype(int)
tp_ma = (ma_anomaly & y_true).sum()
fp_ma = (ma_anomaly & (1 - y_true)).sum()
fn_ma = ((1 - ma_anomaly) & y_true).sum()
p_ma = tp_ma / (tp_ma + fp_ma) if (tp_ma + fp_ma) > 0 else 0
r_ma = tp_ma / (tp_ma + fn_ma) if (tp_ma + fn_ma) > 0 else 0
print(f"移动平均窗口: {window}")
print(f"残差标准差: {residual_std:.4f}")
print(f"检出异常数: {ma_anomaly.sum()}, P={p_ma:.2f}, R={r_ma:.2f}")

# 不同窗口对比
print("\n--- 不同窗口对比 ---")
for w in [3, 5, 7, 14, 30]:
    ma_w = ts.rolling(window=w, center=True).mean()
    res_w = ts - ma_w
    std_w = res_w.std()
    anom_w = (np.abs(res_w) > 3 * std_w).astype(int)
    tp_w = (anom_w & y_true).sum()
    fp_w = (anom_w & (1 - y_true)).sum()
    fn_w = ((1 - anom_w) & y_true).sum()
    p_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
    r_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
    print(f"  window={w}: 检出{anom_w.sum()}, P={p_w:.2f}, R={r_w:.2f}")

# === STL分解异常检测 ===
print("\n=== STL分解异常检测 ===")

# 手动实现简化STL: 趋势+季节+残差
# 趋势: 较大窗口移动平均
trend_stl = ts.rolling(window=30, center=True).mean()
detrended = ts - trend_stl

# 季节: 按周期取均值
period = 365
seasonal = np.zeros(n_days)
for d in range(period):
    mask = np.arange(d, n_days, period)
    if len(mask) > 0:
        seasonal[mask] = detrended.iloc[mask].mean()
seasonal_series = pd.Series(seasonal, index=dates)

# 残差
residual_stl = ts - trend_stl - seasonal_series
residual_stl_clean = residual_stl.dropna()

stl_anomaly = (np.abs(residual_stl) > 3 * residual_stl_clean.std()).astype(int)
tp_stl = (stl_anomaly & y_true).sum()
fp_stl = (stl_anomaly & (1 - y_true)).sum()
fn_stl = ((1 - stl_anomaly) & y_true).sum()
p_stl = tp_stl / (tp_stl + fp_stl) if (tp_stl + fp_stl) > 0 else 0
r_stl = tp_stl / (tp_stl + fn_stl) if (tp_stl + fn_stl) > 0 else 0
print(f"STL残差标准差: {residual_stl_clean.std():.4f}")
print(f"STL检出异常数: {stl_anomaly.sum()}, P={p_stl:.2f}, R={r_stl:.2f}")

# STL分解结果
print(f"\n--- STL分解统计 ---")
print(f"趋势范围: [{trend_stl.min():.2f}, {trend_stl.max():.2f}]")
print(f"季节性范围: [{seasonal_series.min():.2f}, {seasonal_series.max():.2f}]")
print(f"残差范围: [{residual_stl.min():.2f}, {residual_stl.max():.2f}]")

# === 滑动窗口统计异常 ===
print("\n=== 滑动窗口统计异常 ===")

window_size = 14
zscore_anomaly = np.zeros(n_days, dtype=int)
for i in range(window_size, n_days):
    window_data = values[i - window_size:i]
    w_mean = window_data.mean()
    w_std = window_data.std()
    if w_std > 0 and np.abs(values[i] - w_mean) > 3 * w_std:
        zscore_anomaly[i] = 1

tp_zs = (zscore_anomaly & y_true).sum()
fp_zs = (zscore_anomaly & (1 - y_true)).sum()
fn_zs = ((1 - zscore_anomaly) & y_true).sum()
p_zs = tp_zs / (tp_zs + fp_zs) if (tp_zs + fp_zs) > 0 else 0
r_zs = tp_zs / (tp_zs + fn_zs) if (tp_zs + fn_zs) > 0 else 0
print(f"滑动窗口大小: {window_size}, Z-score阈值: 3")
print(f"检出异常数: {zscore_anomaly.sum()}, P={p_zs:.2f}, R={r_zs:.2f}")

# 不同窗口大小对比
print("\n--- 不同滑动窗口对比 ---")
for ws in [7, 14, 21, 30]:
    zs_anom = np.zeros(n_days, dtype=int)
    for i in range(ws, n_days):
        wd = values[i - ws:i]
        wm, wst = wd.mean(), wd.std()
        if wst > 0 and np.abs(values[i] - wm) > 3 * wst:
            zs_anom[i] = 1
    tp_z = (zs_anom & y_true).sum()
    fp_z = (zs_anom & (1 - y_true)).sum()
    fn_z = ((1 - zs_anom) & y_true).sum()
    p_z = tp_z / (tp_z + fp_z) if (tp_z + fp_z) > 0 else 0
    r_z = tp_z / (tp_z + fn_z) if (tp_z + fn_z) > 0 else 0
    print(f"  window={ws}: 检出{zs_anom.sum()}, P={p_z:.2f}, R={r_z:.2f}")

# 三种方法对比
print("\n=== 三种时间序列异常检测方法对比 ===")
print(f"移动平均+残差: P={p_ma:.2f}, R={r_ma:.2f}")
print(f"STL分解:       P={p_stl:.2f}, R={r_stl:.2f}")
print(f"滑动窗口统计:   P={p_zs:.2f}, R={r_zs:.2f}")
print("移动平均: 简单快速, 适合趋势稳定序列")
print("STL分解: 能分离趋势/季节/残差, 更精确")
print("滑动窗口: 自适应局部变化, 对突变敏感")
