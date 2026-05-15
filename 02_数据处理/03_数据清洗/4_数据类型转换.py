# 数据来源: 自建 DataFrame 演示数据

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 100)

# ============================================================
# 一、构建演示数据
# ============================================================
data = {
    "姓名": ["张三", "李四", "王五", "赵六", "钱七"],
    "年龄": ["25", "30", "二十五", "35", "28"],
    "薪资": ["8000.5", "12000", "N/A", "15000", "10k"],
    "入职日期": ["2020-01-15", "2019/06/20", "20210301", "2022-12-01", "2020.03.15"],
    "部门": ["技术部", "市场部", "技术部", "财务部", "技术部"],
    "是否在职": ["是", "否", "是", "是", "是"],
    "成绩": ["85.5", "92.3", "78.1", "invalid", "88.9"],
}
df = pd.DataFrame(data)
print("=== 原始数据 ===")
print(df)
print()
print("=== 原始数据类型 ===")
print(df.dtypes)
print()

# ============================================================
# 二、astype() 基本类型转换
# ============================================================

# 2.1 将字符串列转换为整数 (成功的情况)
print("=== astype() 转换: 部门列 object -> category ===")
df_astype = df.copy()
df_astype["部门"] = df_astype["部门"].astype("category")
print(f"转换后类型: {df_astype['部门'].dtype}")
print(f"分类类别: {df_astype['部门'].cat.categories.tolist()}")
print()

# 2.2 尝试将含非数字的列转为整数 (会报错)
print("=== astype() 转换: 含非数字的年龄列转整数 ===")
try:
    df_astype["年龄"] = df_astype["年龄"].astype(int)
except ValueError as e:
    print(f"转换失败: {e}")
print()

# 2.3 安全的类型转换: 先过滤再转换
print("=== 安全转换: 仅转换纯数字的年龄行 ===")
df_safe = df.copy()
numeric_mask = df_safe["年龄"].str.isnumeric()
df_safe.loc[numeric_mask, "年龄"] = df_safe.loc[numeric_mask, "年龄"].astype(int)
print(df_safe["年龄"])
print(f"年龄列类型: {df_safe['年龄'].dtype}")
print()

# ============================================================
# 三、pd.to_numeric() 数值转换
# ============================================================

# 3.1 默认 errors='raise': 遇到非数字报错
print("=== pd.to_numeric(errors='raise') ===")
try:
    result = pd.to_numeric(df["薪资"])
except ValueError as e:
    print(f"转换失败: {e}")
print()

# 3.2 errors='coerce': 非数字转为 NaN
print("=== pd.to_numeric(errors='coerce') 非数字转NaN ===")
df_numeric = df.copy()
df_numeric["薪资"] = pd.to_numeric(df_numeric["薪资"], errors="coerce")
print(df_numeric["薪资"])
print(f"类型: {df_numeric['薪资'].dtype}")
print()

# 3.3 errors='ignore': 遇到非数字保持原样
print("=== pd.to_numeric(errors='ignore') 保持原样 ===")
result_ignore = pd.to_numeric(df["薪资"], errors="ignore")
print(result_ignore)
print(f"类型: {result_ignore.dtype}")
print()

# 3.4 downcast 参数: 优化数值类型减少内存
print("=== pd.to_numeric(downcast='integer') 优化整数类型 ===")
df_downcast = df.copy()
df_downcast["成绩"] = pd.to_numeric(df_downcast["成绩"], errors="coerce")
print(f"转换后类型: {df_downcast['成绩'].dtype}")

df_downcast["成绩_int"] = pd.to_numeric(df_downcast["成绩"], errors="coerce", downcast="integer")
print(f"downcast后类型: {df_downcast['成绩_int'].dtype}")
print()

# downcast 各选项对比
print("=== downcast 选项对比 ===")
s = pd.Series([1, 2, 3, 4, 5])
for dc in [None, "integer", "signed", "unsigned", "float"]:
    if dc is None:
        result = pd.to_numeric(s)
    else:
        result = pd.to_numeric(s, downcast=dc)
    print(f"  downcast={str(dc):<10} -> dtype: {result.dtype}")
print()

# 3.5 综合示例: 清洗薪资列
print("=== 综合示例: 清洗薪资列 ===")
df_salary = df.copy()
df_salary["薪资"] = pd.to_numeric(
    df_salary["薪资"].str.replace("k", "000", regex=False),
    errors="coerce",
)
print(df_salary["薪资"])
print(f"缺失值数量: {df_salary['薪资'].isnull().sum()}")
print()

# ============================================================
# 四、pd.to_datetime() 日期转换
# ============================================================

# 4.1 自动推断日期格式
print("=== pd.to_datetime() 自动推断 ===")
dates = pd.Series(["2020-01-15", "2019/06/20", "2021-03-01"])
result = pd.to_datetime(dates)
print(result)
print(f"类型: {result.dtype}")
print()

# 4.2 指定 format 参数
print("=== pd.to_datetime(format=...) 指定格式 ===")
dates_compact = pd.Series(["20210301", "20220115", "20200620"])
result_fmt = pd.to_datetime(dates_compact, format="%Y%m%d")
print(result_fmt)
print()

# 4.3 混合格式的日期转换
print("=== 混合格式日期转换 ===")
df_date = df.copy()
df_date["入职日期"] = pd.to_datetime(df_date["入职日期"], errors="coerce")
print(df_date["入职日期"])
print(f"类型: {df_date['入职日期'].dtype}")
print()

# 4.4 errors 参数
print("=== pd.to_datetime(errors='coerce') ===")
bad_dates = pd.Series(["2020-01-15", "not-a-date", "2021-13-01"])
result_coerce = pd.to_datetime(bad_dates, errors="coerce")
print(result_coerce)
print()

# 4.5 日期提取
print("=== 从日期列提取信息 ===")
df_date_extract = df.copy()
df_date_extract["入职日期"] = pd.to_datetime(df_date_extract["入职日期"], errors="coerce")
df_date_extract["入职年份"] = df_date_extract["入职日期"].dt.year
df_date_extract["入职月份"] = df_date_extract["入职日期"].dt.month
df_date_extract["入职日"] = df_date_extract["入职日期"].dt.day
df_date_extract["入职季度"] = df_date_extract["入职日期"].dt.quarter
df_date_extract["星期几"] = df_date_extract["入职日期"].dt.day_name()
print(df_date_extract[["入职日期", "入职年份", "入职月份", "入职日", "入职季度", "星期几"]])
print()

# ============================================================
# 五、category 类型优化内存
# ============================================================

print("=== category 类型内存优化 ===")

# 5.1 构造较大数据集
df_large = pd.DataFrame({
    "城市": np.random.choice(["北京", "上海", "广州", "深圳", "杭州"], 100000),
    "部门": np.random.choice(["技术部", "市场部", "财务部", "人事部"], 100000),
    "薪资": np.random.randint(5000, 30000, 100000),
})

# 5.2 转换前内存
mem_before = df_large.memory_usage(deep=True)
print("转换前内存占用:")
print(mem_before)
print(f"总计: {mem_before.sum() / 1024:.1f} KB")
print()

# 5.3 转换为 category
df_large["城市"] = df_large["城市"].astype("category")
df_large["部门"] = df_large["部门"].astype("category")

mem_after = df_large.memory_usage(deep=True)
print("转换为category后内存占用:")
print(mem_after)
print(f"总计: {mem_after.sum() / 1024:.1f} KB")
print()

# 5.4 内存节省比例
city_saving = (1 - mem_after["城市"] / mem_before["城市"]) * 100
dept_saving = (1 - mem_after["部门"] / mem_before["部门"]) * 100
print(f"城市列内存节省: {city_saving:.1f}%")
print(f"部门列内存节省: {dept_saving:.1f}%")
print()

# 5.5 category 类型的常用操作
print("=== category 类型常用操作 ===")
cat_series = df_large["城市"]
print(f"分类类别: {cat_series.cat.categories.tolist()}")
print(f"类别数量: {cat_series.cat.codes.nunique()}")
print(f"值计数:\n{cat_series.value_counts()}")
print()

# ============================================================
# 六、object 转数值的完整流程
# ============================================================

print("=== object -> 数值 完整转换流程 ===")
df_convert = df.copy()
print(f"转换前数据类型:")
print(df_convert.dtypes)
print()

# 步骤1: 清洗字符串
df_convert["成绩"] = df_convert["成绩"].str.strip()
df_convert["薪资"] = df_convert["薪资"].str.replace("k", "000", regex=False)

# 步骤2: 使用 to_numeric 转换
df_convert["成绩"] = pd.to_numeric(df_convert["成绩"], errors="coerce")
df_convert["薪资"] = pd.to_numeric(df_convert["薪资"], errors="coerce")

# 步骤3: 处理转换后的缺失值
df_convert["成绩"] = df_convert["成绩"].fillna(df_convert["成绩"].median())
df_convert["薪资"] = df_convert["薪资"].fillna(df_convert["薪资"].median())

# 步骤4: 优化类型
df_convert["成绩"] = pd.to_numeric(df_convert["成绩"], downcast="float")
df_convert["薪资"] = pd.to_numeric(df_convert["薪资"], downcast="float")

print(f"转换后数据类型:")
print(df_convert.dtypes)
print()
print(df_convert[["姓名", "成绩", "薪资"]])
