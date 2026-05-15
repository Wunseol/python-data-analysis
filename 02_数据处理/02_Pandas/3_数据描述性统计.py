# 数据来源: mtcars.csv (经典汽车数据集)
# 本脚本演示 Pandas 描述性统计方法
# 注意: 需要确保 mtcars.csv 文件存在于同目录下

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

csv_path = BASE_DIR / 'mtcars.csv'
if not csv_path.exists():
    print(f"未找到 mtcars.csv，将使用自建数据演示")
    data = {
        'model': [f'Car{i}' for i in range(1, 33)],
        'mpg': np.random.uniform(10, 35, 32).round(1),
        'cyl': np.random.choice([4, 6, 8], 32),
        'disp': np.random.uniform(70, 470, 32).round(1),
        'hp': np.random.randint(50, 340, 32),
        'drat': np.random.uniform(2.5, 5.0, 32).round(2),
        'wt': np.random.uniform(1.5, 5.5, 32).round(3),
        'qsec': np.random.uniform(14, 23, 32).round(1),
        'vs': np.random.choice([0, 1], 32),
        'am': np.random.choice([0, 1], 32),
        'gear': np.random.choice([3, 4, 5], 32),
        'carb': np.random.choice([1, 2, 3, 4, 6, 8], 32)
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(csv_path)

print("=" * 60)
print("一、数据概览")
print("=" * 60)

print(f"\n数据形状: {df.shape}")
print(f"\n前5行:\n{df.head()}")
print(f"\n数据类型:\n{df.dtypes}")

print("\n" + "=" * 60)
print("二、describe() 描述性统计")
print("=" * 60)

print("\n默认 describe() (数值列):")
print(df.describe())

print("\ndescribe(include='all'):")
print(df.describe(include='all'))

print("\n仅对特定列描述:")
print(df[['mpg', 'hp', 'wt']].describe())

print("\n" + "=" * 60)
print("三、集中趋势统计量")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns

print(f"\n均值 (mean):\n{df[numeric_cols].mean()}")
print(f"\n中位数 (median):\n{df[numeric_cols].median()}")

print("\n" + "=" * 60)
print("四、离散程度统计量")
print("=" * 60)

print(f"\n标准差 (std):\n{df[numeric_cols].std()}")
print(f"\n方差 (var):\n{df[numeric_cols].var()}")
print(f"\n最小值 (min):\n{df[numeric_cols].min()}")
print(f"\n最大值 (max):\n{df[numeric_cols].max()}")
print(f"\n极差 (range = max - min):\n{df[numeric_cols].max() - df[numeric_cols].min()}")

print("\n" + "=" * 60)
print("五、分位数 quantile()")
print("=" * 60)

print(f"\n25% 分位数:\n{df[numeric_cols].quantile(0.25)}")
print(f"\n50% 分位数 (中位数):\n{df[numeric_cols].quantile(0.5)}")
print(f"\n75% 分位数:\n{df[numeric_cols].quantile(0.75)}")
print(f"\n自定义多个分位数:\n{df[numeric_cols].quantile([0.1, 0.25, 0.5, 0.75, 0.9])}")

print("\n" + "=" * 60)
print("六、累计统计")
print("=" * 60)

mpg = df['mpg'].head(10)
print(f"\n前10行 mpg 数据:\n{mpg.tolist()}")
print(f"\n累计和 (cumsum):\n{mpg.cumsum().tolist()}")
print(f"\n累计最大值 (cummax):\n{mpg.cummax().tolist()}")
print(f"\n累计最小值 (cummin):\n{mpg.cummin().tolist()}")
print(f"\n累计乘积 (cumprod) 前5行:\n{df['mpg'].head(5).cumprod().tolist()}")

print("\n" + "=" * 60)
print("七、idxmin() 和 idxmax()")
print("=" * 60)

print(f"\nmpg 最大值所在索引: {df['mpg'].idxmax()} (值: {df['mpg'].max()})")
print(f"mpg 最小值所在索引: {df['mpg'].idxmin()} (值: {df['mpg'].min()})")
print(f"\nhp 最大值所在索引: {df['hp'].idxmax()} (值: {df['hp'].max()})")
print(f"hp 最小值所在索引: {df['hp'].idxmin()} (值: {df['hp'].min()})")

print("\n" + "=" * 60)
print("八、相关系数 corr() 和协方差 cov()")
print("=" * 60)

selected = df[['mpg', 'hp', 'wt', 'disp']]

print("\n相关系数矩阵 (Pearson):")
print(selected.corr().round(3))

print("\n相关系数矩阵 (Spearman):")
print(selected.corr(method='spearman').round(3))

print("\n协方差矩阵:")
print(selected.cov().round(2))

print("\n两列之间的相关系数:")
print(f"mpg 与 hp 的相关系数: {df['mpg'].corr(df['hp']):.4f}")
print(f"mpg 与 wt 的相关系数: {df['mpg'].corr(df['wt']):.4f}")

print("\n两列之间的协方差:")
print(f"mpg 与 hp 的协方差: {df['mpg'].cov(df['hp']):.2f}")

print("\n" + "=" * 60)
print("九、其他统计方法")
print("=" * 60)

print(f"\n计数 (count):\n{df.count()}")
print(f"\n偏度 (skew):\n{df[numeric_cols].skew().round(3)}")
print(f"\n峰度 (kurtosis):\n{df[numeric_cols].kurtosis().round(3)}")
print(f"\n绝对值 (abs) 示例:\n{pd.Series([-3, 1, -4, 1, -5]).abs()}")
print(f"\n标准误 (sem):\n{df[numeric_cols].sem().round(4)}")
