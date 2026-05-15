# 数据来源: 模拟业务指标时序数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
n_days = len(dates)

trend = np.linspace(5000, 8000, n_days)
seasonal = 500 * np.sin(2 * np.pi * np.arange(n_days) / 30)
noise = np.random.normal(0, 300, n_days)
gmv = trend + seasonal + noise

anomaly_idx = [60, 120, 200, 280]
for idx in anomaly_idx:
    gmv[idx] += np.random.choice([-1, 1]) * np.random.uniform(2000, 3500)

metrics = pd.DataFrame({
    "日期": dates,
    "GMV": gmv.round(2),
    "DAU": (gmv * np.random.uniform(0.05, 0.08, n_days)).astype(int),
    "订单量": (gmv / np.random.uniform(80, 120, n_days)).astype(int),
    "转化率": np.random.uniform(0.02, 0.06, n_days).round(4),
    "客单价": np.random.uniform(80, 150, n_days).round(2),
})

metrics["北极星指标"] = metrics["GMV"]
metrics["一级指标_流量"] = metrics["DAU"]
metrics["一级指标_转化"] = metrics["转化率"]
metrics["一级指标_客单价"] = metrics["客单价"]
metrics["二级指标_新用户占比"] = np.random.uniform(0.2, 0.5, n_days).round(3)
metrics["二级指标_复购率"] = np.random.uniform(0.15, 0.4, n_days).round(3)
metrics["二级指标_人均浏览页数"] = np.random.uniform(3, 8, n_days).round(1)

print("=" * 60)
print("指标体系概览")
print("=" * 60)
print(f"北极星指标: GMV (日均 {metrics['GMV'].mean():.0f})")
print(f"\n一级指标拆解:")
print(f"  流量(DAU): 日均 {metrics['DAU'].mean():.0f}")
print(f"  转化率: 日均 {metrics['转化率'].mean():.2%}")
print(f"  客单价: 日均 {metrics['客单价'].mean():.0f}元")
print(f"\n二级指标:")
print(f"  新用户占比: {metrics['二级指标_新用户占比'].mean():.1%}")
print(f"  复购率: {metrics['二级指标_复购率'].mean():.1%}")
print(f"  人均浏览页数: {metrics['二级指标_人均浏览页数'].mean():.1f}")

def detect_anomaly_zscore(series, threshold=3.0):
    mean_val = series.mean()
    std_val = series.std()
    z_scores = (series - mean_val) / std_val
    anomalies = series[abs(z_scores) > threshold]
    return anomalies, z_scores

def detect_anomaly_3sigma(series):
    mean_val = series.mean()
    std_val = series.std()
    upper = mean_val + 3 * std_val
    lower = mean_val - 3 * std_val
    anomalies = series[(series > upper) | (series < lower)]
    return anomalies, upper, lower

gmv_anomalies_z, z_scores = detect_anomaly_zscore(metrics["GMV"])
gmv_anomalies_3s, upper, lower = detect_anomaly_3sigma(metrics["GMV"])

print("\n" + "=" * 60)
print("GMV异常检测 (Z-score方法)")
print("=" * 60)
for idx in gmv_anomalies_z.index:
    print(f"  {metrics.loc[idx, '日期'].strftime('%Y-%m-%d')}: GMV={metrics.loc[idx, 'GMV']:.0f}, Z={z_scores[idx]:.2f}")

print(f"\n3σ上下界: [{lower:.0f}, {upper:.0f}]")
print(f"3σ异常点数: {len(gmv_anomalies_3s)}")

monitor_data = metrics[["日期", "北极星指标", "一级指标_流量", "一级指标_转化",
                         "一级指标_客单价", "二级指标_新用户占比",
                         "二级指标_复购率", "二级指标_人均浏览页数"]].copy()

print("\n" + "=" * 60)
print("监控看板数据（前5行）")
print("=" * 60)
print(monitor_data.head().to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(metrics["日期"], metrics["GMV"], color="#2196F3", linewidth=1)
axes[0].axhline(upper, color="red", linestyle="--", label=f"3σ上界 ({upper:.0f})")
axes[0].axhline(lower, color="red", linestyle="--", label=f"3σ下界 ({lower:.0f})")
axes[0].fill_between(metrics["日期"], lower, upper, alpha=0.1, color="green")
anomaly_dates = metrics.loc[gmv_anomalies_3s.index, "日期"]
anomaly_vals = gmv_anomalies_3s.values
axes[0].scatter(anomaly_dates, anomaly_vals, color="red", s=50, zorder=5, label="异常点")
axes[0].set_title("GMV监控与异常检测")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(metrics["日期"], z_scores, color="#FF9800", linewidth=1)
axes[1].axhline(3, color="red", linestyle="--", label="Z=3")
axes[1].axhline(-3, color="red", linestyle="--", label="Z=-3")
axes[1].fill_between(metrics["日期"], -3, 3, alpha=0.1, color="green")
axes[1].set_title("Z-score异常检测")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "指标监控.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / '指标监控.png'}")
