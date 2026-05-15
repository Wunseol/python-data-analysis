# 数据来源: seaborn 内置数据集 tips (餐厅小费数据)

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 100)

# ============================================================
# 一、加载数据
# ============================================================
df = sns.load_dataset("tips")
print("=== 原始数据形状 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
print(df.head())
print()

# 注入一些异常值用于演示
df.loc[0, "total_bill"] = 500
df.loc[1, "tip"] = 100
df.loc[2, "total_bill"] = -10
print("=== 注入异常值后的数据 (前5行) ===")
print(df.head())
print()

# ============================================================
# 二、IQR 方法 (四分位距法)
# ============================================================

# 2.1 计算 Q1, Q3, IQR
print("=== IQR 方法检测异常值 ===")
col = "total_bill"
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"列: {col}")
print(f"Q1 (25%分位数): {Q1:.2f}")
print(f"Q3 (75%分位数): {Q3:.2f}")
print(f"IQR (四分位距): {IQR:.2f}")
print(f"下界 (Q1 - 1.5*IQR): {lower_fence:.2f}")
print(f"上界 (Q3 + 1.5*IQR): {upper_fence:.2f}")
print()

# 2.2 识别异常值
iqr_outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
print(f"IQR方法检测到的异常值数量: {len(iqr_outliers)}")
print(iqr_outliers[[col, "tip", "size"]])
print()

# 2.3 对 tip 列同样进行 IQR 检测
print("=== tip 列 IQR 检测 ===")
col_tip = "tip"
Q1_tip = df[col_tip].quantile(0.25)
Q3_tip = df[col_tip].quantile(0.75)
IQR_tip = Q3_tip - Q1_tip
lower_tip = Q1_tip - 1.5 * IQR_tip
upper_tip = Q3_tip + 1.5 * IQR_tip
print(f"tip - 下界: {lower_tip:.2f}, 上界: {upper_tip:.2f}")
tip_outliers = df[(df[col_tip] < lower_tip) | (df[col_tip] > upper_tip)]
print(f"tip列异常值数量: {len(tip_outliers)}")
print()

# 2.4 封装为函数
def detect_outliers_iqr(data, column, factor=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers, lower, upper

print("=== 使用函数检测 total_bill 异常值 (factor=1.5) ===")
outliers, low, high = detect_outliers_iqr(df, "total_bill")
print(f"下界: {low:.2f}, 上界: {high:.2f}")
print(f"异常值数量: {len(outliers)}")
print()

print("=== 使用更严格的 factor=3.0 ===")
outliers_strict, low_s, high_s = detect_outliers_iqr(df, "total_bill", factor=3.0)
print(f"下界: {low_s:.2f}, 上界: {high_s:.2f}")
print(f"异常值数量: {len(outliers_strict)}")
print()

# ============================================================
# 三、Z-Score 方法
# ============================================================

# 3.1 使用 scipy.stats.zscore 计算 Z 分数
print("=== Z-Score 方法检测异常值 ===")
col = "total_bill"
z_scores = stats.zscore(df[col].dropna())
z_scores_df = pd.DataFrame({"value": df[col].dropna(), "z_score": z_scores})
print(z_scores_df.head(10))
print()

# 3.2 以 |Z| > 3 为阈值识别异常值
threshold = 3
z_outliers = z_scores_df[np.abs(z_scores_df["z_score"]) > threshold]
print(f"Z-Score方法 (阈值={threshold}) 检测到的异常值数量: {len(z_outliers)}")
print(z_outliers)
print()

# 3.3 不同阈值的比较
print("=== 不同Z-Score阈值比较 ===")
for t in [2, 2.5, 3]:
    count = (np.abs(z_scores) > t).sum()
    print(f"  阈值 |Z| > {t}: 异常值数量 = {count}")
print()

# 3.4 封装为函数
def detect_outliers_zscore(data, column, threshold=3):
    z_scores = stats.zscore(data[column].dropna())
    mask = np.abs(z_scores) > threshold
    outlier_indices = data[column].dropna().index[mask]
    return data.loc[outlier_indices]

print("=== 使用函数检测 tip 列异常值 ===")
z_tip_outliers = detect_outliers_zscore(df, "tip", threshold=3)
print(f"Z-Score方法检测tip异常值数量: {len(z_tip_outliers)}")
print()

# ============================================================
# 四、箱线图可视化
# ============================================================

print("=== 箱线图统计信息 ===")
col = "total_bill"
desc = df[col].describe()
print(desc)
print()

# 使用 matplotlib 绘制箱线图 (保存到文件)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].boxplot(df["total_bill"].dropna())
axes[0].set_title("total_bill 箱线图")
axes[0].set_ylabel("金额")

axes[1].boxplot(df["tip"].dropna())
axes[1].set_title("tip 箱线图")
axes[1].set_ylabel("金额")

axes[2].boxplot([df["total_bill"].dropna(), df["tip"].dropna()], labels=["total_bill", "tip"])
axes[2].set_title("total_bill vs tip")
axes[2].set_ylabel("金额")

plt.tight_layout()
output_path = Path(__file__).parent / "outlier_boxplot.png"
plt.savefig(output_path, dpi=100)
plt.close()
print(f"箱线图已保存到: {output_path}")
print()

# ============================================================
# 五、异常值处理 — 截断法 (Winsorization / Capping)
# ============================================================

print("=== 截断法处理异常值 ===")
col = "total_bill"
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_capped = df.copy()
df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)

print(f"截断前 - 最小值: {df[col].min():.2f}, 最大值: {df[col].max():.2f}")
print(f"截断后 - 最小值: {df_capped[col].min():.2f}, 最大值: {df_capped[col].max():.2f}")
print(f"截断下界: {lower:.2f}, 截断上界: {upper:.2f}")
print()

# 使用百分位数截断
print("=== 百分位数截断 (1% ~ 99%) ===")
p1 = df[col].quantile(0.01)
p99 = df[col].quantile(0.99)
df_capped_pct = df.copy()
df_capped_pct[col] = df_capped_pct[col].clip(lower=p1, upper=p99)
print(f"1%分位数: {p1:.2f}, 99%分位数: {p99:.2f}")
print(f"截断前 - 最小值: {df[col].min():.2f}, 最大值: {df[col].max():.2f}")
print(f"截断后 - 最小值: {df_capped_pct[col].min():.2f}, 最大值: {df_capped_pct[col].max():.2f}")
print()

# ============================================================
# 六、异常值处理 — 中位数替换
# ============================================================

print("=== 中位数替换异常值 ===")
col = "total_bill"
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
median_val = df[col].median()

df_median_replace = df.copy()
outlier_mask = (df_median_replace[col] < lower) | (df_median_replace[col] > upper)
df_median_replace.loc[outlier_mask, col] = median_val

print(f"中位数: {median_val:.2f}")
print(f"被替换的异常值数量: {outlier_mask.sum()}")
print(f"替换前 - 均值: {df[col].mean():.2f}, 标准差: {df[col].std():.2f}")
print(f"替换后 - 均值: {df_median_replace[col].mean():.2f}, 标准差: {df_median_replace[col].std():.2f}")
print()

# ============================================================
# 七、异常值处理方法对比
# ============================================================

print("=== 异常值处理方法对比 ===")
col = "total_bill"
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
median_val = df[col].median()

# 各方法处理
df_drop = df[(df[col] >= lower) & (df[col] <= upper)].copy()
df_clip = df.copy()
df_clip[col] = df_clip[col].clip(lower=lower, upper=upper)
df_med = df.copy()
outlier_mask = (df_med[col] < lower) | (df_med[col] > upper)
df_med.loc[outlier_mask, col] = median_val

print(f"{'方法':<15} {'行数':<8} {'均值':<10} {'中位数':<10} {'标准差':<10}")
print("-" * 55)
print(f"{'原始数据':<15} {len(df):<8} {df[col].mean():<10.2f} {df[col].median():<10.2f} {df[col].std():<10.2f}")
print(f"{'删除异常值':<15} {len(df_drop):<8} {df_drop[col].mean():<10.2f} {df_drop[col].median():<10.2f} {df_drop[col].std():<10.2f}")
print(f"{'截断法':<15} {len(df_clip):<8} {df_clip[col].mean():<10.2f} {df_clip[col].median():<10.2f} {df_clip[col].std():<10.2f}")
print(f"{'中位数替换':<15} {len(df_med):<8} {df_med[col].mean():<10.2f} {df_med[col].median():<10.2f} {df_med[col].std():<10.2f}")
