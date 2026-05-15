# 数据来源: seaborn 内置数据集 titanic (泰坦尼克号乘客数据)

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from pathlib import Path

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

# ============================================================
# 一、加载数据与初始评估
# ============================================================
df = sns.load_dataset("titanic")
print("=" * 60)
print("泰坦尼克号数据集综合清洗案例")
print("=" * 60)
print()

print("=== 1.1 原始数据概览 ===")
print(f"形状: {df.shape}")
print()
print("前5行:")
print(df.head())
print()
print("数据类型:")
print(df.dtypes)
print()
print("基本统计:")
print(df.describe(include="all"))
print()

# ============================================================
# 二、清洗前数据质量报告
# ============================================================

def data_quality_report(data, stage="清洗前"):
    print(f"\n{'=' * 40}")
    print(f"  数据质量报告 — {stage}")
    print(f"{'=' * 40}")

    print(f"\n总行数: {len(data)}, 总列数: {len(data.columns)}")

    # 缺失值
    missing = data.isnull().sum()
    missing_pct = (data.isnull().mean() * 100).round(2)
    missing_info = pd.DataFrame({"缺失数": missing, "缺失比例(%)": missing_pct})
    missing_info = missing_info[missing_info["缺失数"] > 0]
    if len(missing_info) > 0:
        print("\n缺失值:")
        print(missing_info)
    else:
        print("\n缺失值: 无")

    # 重复值
    dup_count = data.duplicated().sum()
    print(f"\n完全重复行数: {dup_count}")

    # 数据类型
    print("\n数据类型:")
    for col in data.columns:
        print(f"  {col}: {data[col].dtype}")

    # 内存占用
    mem = data.memory_usage(deep=True).sum() / 1024
    print(f"\n内存占用: {mem:.1f} KB")

    return missing_info

missing_before = data_quality_report(df, "清洗前")

# ============================================================
# 三、步骤1: 缺失值处理
# ============================================================
print("\n" + "=" * 60)
print("步骤1: 缺失值处理")
print("=" * 60)

df_clean = df.copy()

# 3.1 age 列 — 用中位数填充 (按性别分组)
print("\n--- age列: 按性别分组中位数填充 ---")
print(f"填充前缺失数: {df_clean['age'].isnull().sum()}")
age_median_by_sex = df_clean.groupby("sex")["age"].transform("median")
df_clean["age"] = df_clean["age"].fillna(age_median_by_sex)
print(f"填充后缺失数: {df_clean['age'].isnull().sum()}")
print(f"填充后age统计: 均值={df_clean['age'].mean():.2f}, 中位数={df_clean['age'].median():.2f}")

# 3.2 embarked / embark_town 列 — 用众数填充
print("\n--- embarked/embark_town列: 众数填充 ---")
embarked_mode = df_clean["embarked"].mode()[0]
embark_town_mode = df_clean["embark_town"].mode()[0]
print(f"embarked众数: {embarked_mode}")
print(f"embark_town众数: {embark_town_mode}")
df_clean["embarked"] = df_clean["embarked"].fillna(embarked_mode)
df_clean["embark_town"] = df_clean["embark_town"].fillna(embark_town_mode)
print(f"填充后embarked缺失数: {df_clean['embarked'].isnull().sum()}")
print(f"填充后embark_town缺失数: {df_clean['embark_town'].isnull().sum()}")

# 3.3 deck 列 — 缺失比例过高, 新增'Unknown'类别
print("\n--- deck列: 缺失比例过高, 填充'Unknown' ---")
deck_missing_pct = df_clean["deck"].isnull().mean() * 100
print(f"deck缺失比例: {deck_missing_pct:.1f}%")
df_clean["deck"] = df_clean["deck"].fillna("Unknown")
print(f"填充后deck缺失数: {df_clean['deck'].isnull().sum()}")

print(f"\n步骤1完成后总缺失值: {df_clean.isnull().sum().sum()}")

# ============================================================
# 四、步骤2: 重复值处理
# ============================================================
print("\n" + "=" * 60)
print("步骤2: 重复值处理")
print("=" * 60)

dup_count = df_clean.duplicated().sum()
print(f"完全重复行数: {dup_count}")

if dup_count > 0:
    print("重复行示例:")
    print(df_clean[df_clean.duplicated(keep=False)].head(10))
    df_clean = df_clean.drop_duplicates()
    print(f"删除重复行后行数: {len(df_clean)}")
else:
    print("无完全重复行")

# 检查关键字段重复
key_cols = ["sex", "age", "fare", "embarked"]
key_dup = df_clean.duplicated(subset=key_cols).sum()
print(f"关键字段({', '.join(key_cols)})重复行数: {key_dup}")

# ============================================================
# 五、步骤3: 异常值处理
# ============================================================
print("\n" + "=" * 60)
print("步骤3: 异常值处理")
print("=" * 60)

# 5.1 fare 列异常值检测
print("\n--- fare列异常值检测 ---")
col = "fare"
Q1 = df_clean[col].quantile(0.25)
Q3 = df_clean[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"下界={lower:.2f}, 上界={upper:.2f}")

iqr_outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
print(f"IQR方法异常值数量: {len(iqr_outliers)}")

# Z-Score 检测
z_scores = stats.zscore(df_clean[col].dropna())
z_outliers = np.abs(z_scores) > 3
print(f"Z-Score方法(|Z|>3)异常值数量: {z_outliers.sum()}")

# 5.2 异常值处理: 截断法
print("\n--- fare列异常值处理: 截断法 ---")
df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
print(f"处理后 - 最小值: {df_clean[col].min():.2f}, 最大值: {df_clean[col].max():.2f}")

# 5.3 age 列异常值检测
print("\n--- age列异常值检测 ---")
col_age = "age"
Q1_age = df_clean[col_age].quantile(0.25)
Q3_age = df_clean[col_age].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age
age_outliers = df_clean[(df_clean[col_age] < lower_age) | (df_clean[col_age] > upper_age)]
print(f"age异常值数量: {len(age_outliers)}")
print(f"age范围: [{df_clean[col_age].min():.1f}, {df_clean[col_age].max():.1f}]")

# ============================================================
# 六、步骤4: 数据类型转换
# ============================================================
print("\n" + "=" * 60)
print("步骤4: 数据类型转换")
print("=" * 60)

print("\n转换前数据类型:")
print(df_clean.dtypes)

# 6.1 category 类型优化
cat_cols = ["sex", "embarked", "embark_town", "class", "who", "adult_male", "deck", "alive", "alone"]
for col in cat_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype("category")

# 6.2 数值类型优化
df_clean["survived"] = df_clean["survived"].astype("int8")
df_clean["pclass"] = df_clean["pclass"].astype("int8")
df_clean["sibsp"] = df_clean["sibsp"].astype("int8")
df_clean["parch"] = df_clean["parch"].astype("int8")

# 6.3 age 转换为 float32
df_clean["age"] = df_clean["age"].astype("float32")
df_clean["fare"] = df_clean["fare"].astype("float32")

print("\n转换后数据类型:")
print(df_clean.dtypes)

# 内存对比
mem_before_kb = df.memory_usage(deep=True).sum() / 1024
mem_after_kb = df_clean.memory_usage(deep=True).sum() / 1024
print(f"\n内存优化: {mem_before_kb:.1f} KB -> {mem_after_kb:.1f} KB (节省 {(1 - mem_after_kb/mem_before_kb)*100:.1f}%)")

# ============================================================
# 七、步骤5: 字符串清洗
# ============================================================
print("\n" + "=" * 60)
print("步骤5: 字符串清洗")
print("=" * 60)

# 7.1 who 列标准化
print("\n--- who列值分布 ---")
print(df_clean["who"].value_counts())

# 7.2 embark_town 列标准化 (去除首尾空白, 统一大小写)
if df_clean["embark_town"].dtype == "category":
    df_clean["embark_town"] = df_clean["embark_town"].str.strip()
    print(f"embark_town唯一值: {df_clean['embark_town'].unique().tolist()}")

# 7.3 sex 列标准化
print(f"sex唯一值: {df_clean['sex'].unique().tolist()}")

# 7.4 deck 列标准化
print(f"deck唯一值: {df_clean['deck'].unique().tolist()}")

# ============================================================
# 八、步骤6: 数据验证
# ============================================================
print("\n" + "=" * 60)
print("步骤6: 数据验证")
print("=" * 60)

# 8.1 缺失值验证
print("\n--- 缺失值验证 ---")
remaining_missing = df_clean.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if len(remaining_missing) > 0:
    print(f"仍有缺失值的列:\n{remaining_missing}")
else:
    print("所有缺失值已处理完毕!")

# 8.2 重复值验证
print("\n--- 重复值验证 ---")
remaining_dup = df_clean.duplicated().sum()
print(f"剩余重复行数: {remaining_dup}")

# 8.3 数值范围验证
print("\n--- 数值范围验证 ---")
print(f"age范围: [{df_clean['age'].min():.1f}, {df_clean['age'].max():.1f}]")
print(f"fare范围: [{df_clean['fare'].min():.2f}, {df_clean['fare'].max():.2f}]")
print(f"survived唯一值: {sorted(df_clean['survived'].unique().tolist())}")
print(f"pclass唯一值: {sorted(df_clean['pclass'].unique().tolist())}")

# 8.4 类别值验证
print("\n--- 类别值验证 ---")
for col in cat_cols:
    if col in df_clean.columns:
        n_unique = df_clean[col].nunique()
        print(f"  {col}: {n_unique} 个唯一值")

# ============================================================
# 九、清洗前后对比
# ============================================================
print("\n" + "=" * 60)
print("清洗前后对比")
print("=" * 60)

print(f"\n{'指标':<20} {'清洗前':<20} {'清洗后':<20}")
print("-" * 60)
print(f"{'行数':<20} {len(df):<20} {len(df_clean):<20}")
print(f"{'列数':<20} {len(df.columns):<20} {len(df_clean.columns):<20}")
print(f"{'缺失值总数':<20} {df.isnull().sum().sum():<20} {df_clean.isnull().sum().sum():<20}")
print(f"{'重复行数':<20} {df.duplicated().sum():<20} {df_clean.duplicated().sum():<20}")
print(f"{'内存占用(KB)':<20} {mem_before_kb:<20.1f} {mem_after_kb:<20.1f}")

print("\n清洗前各列缺失值:")
for col in df.columns:
    miss = df[col].isnull().sum()
    if miss > 0:
        print(f"  {col}: {miss} ({miss/len(df)*100:.1f}%)")

print("\n清洗后各列缺失值:")
for col in df_clean.columns:
    miss = df_clean[col].isnull().sum()
    if miss > 0:
        print(f"  {col}: {miss} ({miss/len(df_clean)*100:.1f}%)")

# ============================================================
# 十、保存清洗后数据
# ============================================================
output_path = Path(__file__).parent / "titanic_cleaned.csv"
df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n清洗后数据已保存到: {output_path}")

print("\n" + "=" * 60)
print("综合清洗完成!")
print("=" * 60)
print("\n清洗流程总结:")
print("  1. 缺失值处理: age按性别中位数填充, embarked众数填充, deck填'Unknown'")
print("  2. 重复值处理: 删除完全重复行")
print("  3. 异常值处理: fare列IQR截断法")
print("  4. 类型转换: category优化, 数值类型降级")
print("  5. 字符串清洗: 去除空白, 统一格式")
print("  6. 数据验证: 确认缺失值、重复值、数值范围")
