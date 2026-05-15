# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, seaborn>=0.13
# 数据来源: seaborn 内置数据集 titanic (泰坦尼克号乘客数据)

import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 100)

# ============================================================
# 一、加载数据
# ============================================================
df = sns.load_dataset("titanic")
print("=== 原始数据形状 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
print()

# ============================================================
# 二、缺失值检测
# ============================================================

# 2.1 isnull / isna / notnull 基本检测
print("=== isnull() 检测缺失值 (前5行) ===")
print(df.isnull().head())
print()

print("=== isna() 检测缺失值 (前5行) ===")
print(df.isna().head())
print()

print("=== notnull() 检测非缺失值 (前5行) ===")
print(df.notnull().head())
print()

# 2.2 每列缺失值计数
print("=== 每列缺失值计数 ===")
missing_count = df.isnull().sum()
print(missing_count)
print()

# 2.3 每列缺失值百分比
print("=== 每列缺失值百分比 ===")
missing_pct = df.isnull().mean() * 100
missing_pct = missing_pct.round(2)
print(missing_pct)
print()

# 2.4 汇总缺失值信息
print("=== 缺失值汇总表 ===")
missing_summary = pd.DataFrame({
    "缺失数量": df.isnull().sum(),
    "缺失比例(%)": (df.isnull().mean() * 100).round(2),
    "非缺失数量": df.notnull().sum(),
    "数据类型": df.dtypes,
})
print(missing_summary)
print()

# 2.5 筛选含有缺失值的行
print("=== 含有缺失值的行 (前5行) ===")
rows_with_missing = df[df.isnull().any(axis=1)]
print(f"含有缺失值的行数: {len(rows_with_missing)}")
print(rows_with_missing.head())
print()

# ============================================================
# 三、删除缺失值 — dropna
# ============================================================

# 3.1 删除含有任何缺失值的行
print("=== dropna() 删除含缺失值的行 ===")
df_drop_all = df.dropna()
print(f"原始行数: {len(df)}, 删除后行数: {len(df_drop_all)}")
print()

# 3.2 axis 参数: 按列删除
print("=== dropna(axis=1) 删除含缺失值的列 ===")
df_drop_cols = df.dropna(axis=1)
print(f"原始列数: {df.shape[1]}, 删除后列数: {df_drop_cols.shape[1]}")
print(f"被删除的列: {list(set(df.columns) - set(df_drop_cols.columns))}")
print()

# 3.3 how 参数: 'all' 仅删除全为缺失值的行
print("=== dropna(how='all') 仅删除全为缺失值的行 ===")
df_drop_all_na = df.dropna(how="all")
print(f"原始行数: {len(df)}, 删除后行数: {len(df_drop_all_na)}")
print()

# 3.4 thresh 参数: 保留至少有 thresh 个非缺失值的行
print("=== dropna(thresh=14) 保留至少14个非缺失值的行 ===")
df_drop_thresh = df.dropna(thresh=14)
print(f"原始行数: {len(df)}, 删除后行数: {len(df_drop_thresh)}")
print()

# 3.5 subset 参数: 仅对指定列检查缺失值
print("=== dropna(subset=['age']) 仅根据age列删除行 ===")
df_drop_subset = df.dropna(subset=["age"])
print(f"原始行数: {len(df)}, 删除后行数: {len(df_drop_subset)}")
print()

# ============================================================
# 四、填充缺失值 — fillna
# ============================================================

# 4.1 用常量填充
print("=== fillna(常量) 用常量填充 ===")
df_fill_const = df.copy()
df_fill_const["age"] = df_fill_const["age"].fillna(0)
df_fill_const["embarked"] = df_fill_const["embarked"].fillna("未知")
print(f"age列缺失数: {df_fill_const['age'].isnull().sum()}")
print(f"embarked列缺失数: {df_fill_const['embarked'].isnull().sum()}")
print()

# 4.2 用均值填充
print("=== fillna(均值) 用均值填充age ===")
df_fill_mean = df.copy()
age_mean = df_fill_mean["age"].mean()
df_fill_mean["age"] = df_fill_mean["age"].fillna(age_mean)
print(f"age均值: {age_mean:.2f}")
print(f"填充后age列缺失数: {df_fill_mean['age'].isnull().sum()}")
print()

# 4.3 用中位数填充
print("=== fillna(中位数) 用中位数填充age ===")
df_fill_median = df.copy()
age_median = df_fill_median["age"].median()
df_fill_median["age"] = df_fill_median["age"].fillna(age_median)
print(f"age中位数: {age_median:.2f}")
print(f"填充后age列缺失数: {df_fill_median['age'].isnull().sum()}")
print()

# 4.4 用众数填充
print("=== fillna(众数) 用众数填充embarked ===")
df_fill_mode = df.copy()
embarked_mode = df_fill_mode["embarked"].mode()[0]
df_fill_mode["embarked"] = df_fill_mode["embarked"].fillna(embarked_mode)
print(f"embarked众数: {embarked_mode}")
print(f"填充后embarked列缺失数: {df_fill_mode['embarked'].isnull().sum()}")
print()

# 4.5 前向填充 ffill
print("=== fillna(method='ffill') 前向填充 ===")
df_ffill = df.copy()
df_ffill["age"] = df_ffill["age"].ffill()
print(f"前向填充后age列缺失数: {df_ffill['age'].isnull().sum()}")
print()

# 4.6 后向填充 bfill
print("=== fillna(method='bfill') 后向填充 ===")
df_bfill = df.copy()
df_bfill["age"] = df_bfill["age"].bfill()
print(f"后向填充后age列缺失数: {df_bfill['age'].isnull().sum()}")
print()

# 4.7 不同列使用不同填充策略
print("=== 多列不同策略填充 ===")
df_fill_multi = df.copy()
df_fill_multi["age"] = df_fill_multi["age"].fillna(df_fill_multi["age"].median())
df_fill_multi["embarked"] = df_fill_multi["embarked"].fillna(df_fill_multi["embarked"].mode()[0])
df_fill_multi["embark_town"] = df_fill_multi["embark_town"].fillna("未知")
df_fill_multi["deck"] = df_fill_multi["deck"].fillna("Unknown")
print(f"填充后缺失值统计:")
print(df_fill_multi.isnull().sum())
print()

# ============================================================
# 五、插值法填充 — interpolate
# ============================================================

# 构造含有缺失值的数值序列用于演示
s = pd.Series([1, np.nan, np.nan, 4, 5, np.nan, 7, 8, np.nan, 10])
print("=== 原始含缺失值序列 ===")
print(s)
print()

# 5.1 线性插值 (默认)
print("=== interpolate() 线性插值 ===")
s_linear = s.interpolate(method="linear")
print(s_linear)
print()

# 5.2 多项式插值
print("=== interpolate(method='polynomial', order=2) 二次多项式插值 ===")
s_poly = s.interpolate(method="polynomial", order=2)
print(s_poly)
print()

# 5.3 对 age 列使用插值
print("=== 对age列使用线性插值 ===")
df_interp = df.copy()
df_interp["age"] = df_interp["age"].interpolate(method="linear")
print(f"插值后age列缺失数: {df_interp['age'].isnull().sum()}")
print(f"插值后age列统计:")
print(df_interp["age"].describe())
print()

# ============================================================
# 六、缺失值处理策略选择建议
# ============================================================
print("=== 缺失值处理策略选择建议 ===")
print("1. 缺失比例 < 5%:  可考虑删除行 (dropna)")
print("2. 缺失比例 5%~30%: 推荐填充 (均值/中位数/众数/插值)")
print("3. 缺失比例 > 30%:  考虑删除列或使用高级建模方法")
print("4. 数值型数据: 均值/中位数/插值")
print("5. 分类型数据: 众数/新增'未知'类别")
print("6. 时间序列数据: ffill/bfill/插值")
