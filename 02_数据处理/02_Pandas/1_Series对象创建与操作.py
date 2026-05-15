# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24
# 数据来源: 自建示例数据
# 本脚本演示 Pandas Series 对象的创建与基本操作

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、Series 对象的创建")
print("=" * 60)

# 1. 从列表创建 Series
data_list = [85, 90, 78, 92, 88]
s_from_list = pd.Series(data_list)
print("\n从列表创建 Series:")
print(s_from_list)

# 指定索引
s_with_index = pd.Series(data_list, index=['语文', '数学', '英语', '物理', '化学'], name='成绩')
print("\n指定索引和名称:")
print(s_with_index)

# 2. 从字典创建 Series
data_dict = {'苹果': 5.5, '香蕉': 3.2, '橙子': 4.8, '葡萄': 8.0}
s_from_dict = pd.Series(data_dict)
print("\n从字典创建 Series:")
print(s_from_dict)

# 3. 从标量值创建 Series
s_from_scalar = pd.Series(10, index=['a', 'b', 'c', 'd'])
print("\n从标量值创建 Series:")
print(s_from_scalar)

# 4. 从 NumPy 数组创建 Series
arr = np.array([1.1, 2.2, 3.3, 4.4])
s_from_ndarray = pd.Series(arr, index=['x', 'y', 'z', 'w'])
print("\n从 NumPy 数组创建 Series:")
print(s_from_ndarray)

print("\n" + "=" * 60)
print("二、Series 基本属性")
print("=" * 60)

s = pd.Series([85, 90, 78, 92, 88], index=['语文', '数学', '英语', '物理', '化学'], name='成绩')

print(f"\n索引 (index): {s.index.tolist()}")
print(f"值 (values): {s.values}")
print(f"数据类型 (dtype): {s.dtype}")
print(f"名称 (name): {s.name}")
print(f"形状 (shape): {s.shape}")
print(f"元素数量 (size): {s.size}")
print(f"维度 (ndim): {s.ndim}")

print("\n" + "=" * 60)
print("三、Series 切片与索引")
print("=" * 60)

# 标签索引
print(f"\n标签索引 s['数学']: {s['数学']}")
print(f"标签索引 s[['语文', '英语']]:\n{s[['语文', '英语']]}")

# 位置索引
print(f"\n位置索引 s[0]: {s[0]}")
print(f"位置索引 s[1:3]:\n{s[1:3]}")

# 切片（标签切片包含末端）
print(f"\n标签切片 s['语文':'英语']:\n{s['语文':'英语']}")

print("\n" + "=" * 60)
print("四、布尔索引")
print("=" * 60)

# 比较运算生成布尔 Series
mask = s > 85
print(f"\n布尔条件 s > 85:\n{mask}")
print(f"\n布尔索引筛选:\n{s[mask]}")

# 组合条件
mask_combined = (s >= 80) & (s <= 90)
print(f"\n组合条件 (s >= 80) & (s <= 90):\n{s[mask_combined]}")

print("\n" + "=" * 60)
print("五、isin() 方法")
print("=" * 60)

selected = s.isin([85, 92])
print(f"\ns.isin([85, 92]):\n{selected}")
print(f"\n筛选结果:\n{s[selected]}")

print("\n" + "=" * 60)
print("六、Series 基本数学运算")
print("=" * 60)

s1 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

print(f"\ns1:\n{s1}")
print(f"\ns2:\n{s2}")

# 算术运算
print(f"\ns1 + s2:\n{s1 + s2}")
print(f"\ns1 * 2:\n{s1 * 2}")
print(f"\ns1 ** 2:\n{s1 ** 2}")

# 统计方法
print(f"\ns1.sum(): {s1.sum()}")
print(f"s1.mean(): {s1.mean()}")
print(f"s1.median(): {s1.median()}")
print(f"s1.std(): {s1.std():.4f}")
print(f"s1.min(): {s1.min()}")
print(f"s1.max(): {s1.max()}")
print(f"s1.cumsum():\n{s1.cumsum()}")

# 索引不对齐时的运算
s3 = pd.Series([100, 200], index=['a', 'e'])
print(f"\ns1 + s3 (索引不对齐):\n{s1 + s3}")

# 使用 fill_value 处理缺失值
print(f"\ns1.add(s3, fill_value=0):\n{s1.add(s3, fill_value=0)}")

print("\n" + "=" * 60)
print("七、Series 其他常用方法")
print("=" * 60)

s_dup = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
print(f"\n原 Series: {s_dup.tolist()}")
print(f"唯一值 (unique): {s_dup.unique()}")
print(f"值计数 (value_counts):\n{s_dup.value_counts()}")
print(f"排序 (sort_values):\n{s_dup.sort_values()}")
print(f"降序排序:\n{s_dup.sort_values(ascending=False)}")
print(f"排名 (rank):\n{s_dup.rank()}")
print(f"是否为空 (isna):\n{pd.Series([1, np.nan, 3, None]).isna()}")
