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
    "姓名": ["张三", "李四", "王五", "张三", "赵六", "李四", "张三", "钱七", "孙八", "李四"],
    "年龄": [25, 30, 25, 25, 35, 30, 25, 28, 40, 30],
    "城市": ["北京", "上海", "北京", "北京", "广州", "上海", "北京", "深圳", "杭州", "上海"],
    "薪资": [8000, 12000, 8000, 8000, 15000, 12000, 8000, 10000, 18000, 12000],
    "部门": ["技术部", "市场部", "技术部", "技术部", "财务部", "市场部", "技术部", "人事部", "技术部", "市场部"],
}
df = pd.DataFrame(data)
print("=== 原始数据 ===")
print(df)
print()

# ============================================================
# 二、重复值检测 — duplicated()
# ============================================================

# 2.1 检测完全重复的行
print("=== duplicated() 检测完全重复行 ===")
dup_mask = df.duplicated()
print(dup_mask)
print(f"完全重复行数: {dup_mask.sum()}")
print()

# 2.2 查看重复的行
print("=== 查看重复的行 ===")
print(df[dup_mask])
print()

# 2.3 查看原始行与重复行
print("=== 重复行的原始行与重复行 ===")
dup_rows = df[df.duplicated(keep=False)]
print(dup_rows)
print()

# ============================================================
# 三、keep 参数详解
# ============================================================

# 3.1 keep='first' (默认): 第一次出现标记为False, 后续标记为True
print("=== duplicated(keep='first') 默认行为 ===")
print(df.duplicated(keep="first"))
print(f"标记为重复的行数: {df.duplicated(keep='first').sum()}")
print()

# 3.2 keep='last': 最后一次出现标记为False, 之前标记为True
print("=== duplicated(keep='last') ===")
print(df.duplicated(keep="last"))
print(f"标记为重复的行数: {df.duplicated(keep='last').sum()}")
print()

# 3.3 keep=False: 所有重复行都标记为True
print("=== duplicated(keep=False) 标记所有重复行 ===")
print(df.duplicated(keep=False))
print(f"标记为重复的行数: {df.duplicated(keep=False).sum()}")
print()

# ============================================================
# 四、subset 参数 — 按指定列检测重复
# ============================================================

# 4.1 按单列检测重复
print("=== duplicated(subset=['姓名']) 按姓名检测重复 ===")
name_dup = df.duplicated(subset=["姓名"])
print(name_dup)
print(f"姓名重复的行数: {name_dup.sum()}")
print()

# 4.2 按多列组合检测重复
print("=== duplicated(subset=['姓名', '城市']) 按姓名+城市检测重复 ===")
name_city_dup = df.duplicated(subset=["姓名", "城市"])
print(name_city_dup)
print(f"姓名+城市重复的行数: {name_city_dup.sum()}")
print()

# 4.3 subset + keep 组合
print("=== duplicated(subset=['姓名'], keep=False) ===")
name_dup_all = df.duplicated(subset=["姓名"], keep=False)
print(name_dup_all)
print(f"姓名有重复的行数: {name_dup_all.sum()}")
print()

# ============================================================
# 五、删除重复值 — drop_duplicates()
# ============================================================

# 5.1 删除完全重复的行 (默认 keep='first')
print("=== drop_duplicates() 删除完全重复行 ===")
df_dedup = df.drop_duplicates()
print(f"原始行数: {len(df)}, 去重后行数: {len(df_dedup)}")
print(df_dedup)
print()

# 5.2 keep='first': 保留第一次出现
print("=== drop_duplicates(keep='first') 保留第一次出现 ===")
df_keep_first = df.drop_duplicates(keep="first")
print(df_keep_first)
print()

# 5.3 keep='last': 保留最后一次出现
print("=== drop_duplicates(keep='last') 保留最后一次出现 ===")
df_keep_last = df.drop_duplicates(keep="last")
print(df_keep_last)
print()

# 5.4 keep=False: 删除所有重复行
print("=== drop_duplicates(keep=False) 删除所有重复行 ===")
df_keep_none = df.drop_duplicates(keep=False)
print(f"删除所有重复行后行数: {len(df_keep_none)}")
print(df_keep_none)
print()

# 5.5 按指定列去重
print("=== drop_duplicates(subset=['姓名']) 按姓名去重 ===")
df_dedup_name = df.drop_duplicates(subset=["姓名"])
print(f"按姓名去重后行数: {len(df_dedup_name)}")
print(df_dedup_name)
print()

# 5.6 按多列组合去重
print("=== drop_duplicates(subset=['姓名', '部门']) 按姓名+部门去重 ===")
df_dedup_name_dept = df.drop_duplicates(subset=["姓名", "部门"])
print(f"按姓名+部门去重后行数: {len(df_dedup_name_dept)}")
print(df_dedup_name_dept)
print()

# 5.7 inplace 参数
print("=== inplace=True 原地修改 ===")
df_inplace = df.copy()
df_inplace.drop_duplicates(inplace=True)
print(f"原地去重后行数: {len(df_inplace)}")
print()

# 5.8 ignore_index 参数
print("=== drop_duplicates(ignore_index=True) 重置索引 ===")
df_reset_idx = df.drop_duplicates(ignore_index=True)
print(df_reset_idx)
print()

# ============================================================
# 六、识别重复模式
# ============================================================

print("=== 重复模式分析 ===")

# 6.1 每列的重复值数量
print("--- 各列重复值统计 ---")
for col in df.columns:
    dup_count = df.duplicated(subset=[col]).sum()
    unique_count = df[col].nunique()
    print(f"  {col}: 唯一值={unique_count}, 重复行数={dup_count}")
print()

# 6.2 找出重复最多的值
print("--- 各列重复最多的值 ---")
for col in df.columns:
    value_counts = df[col].value_counts()
    most_common = value_counts.index[0]
    most_common_count = value_counts.iloc[0]
    print(f"  {col}: 最常见值='{most_common}', 出现{most_common_count}次")
print()

# 6.3 重复行占比
total_rows = len(df)
complete_dup_rows = df.duplicated().sum()
subset_dup_rows = df.duplicated(subset=["姓名"]).sum()
print(f"总行数: {total_rows}")
print(f"完全重复行: {complete_dup_rows} ({complete_dup_rows/total_rows*100:.1f}%)")
print(f"按姓名重复行: {subset_dup_rows} ({subset_dup_rows/total_rows*100:.1f}%)")
