"""
案例6：NumPy 数组索引与切片
==============================
本案例演示 NumPy 数组的各种索引与切片操作，包括：
1. 一维数组索引
2. 二维数组索引
3. 布尔索引
4. 花式索引
5. 带步长的切片
6. 负数索引
7. np.where() 条件索引
8. np.take() 按索引取值

数据来源：自行构造的示例数据
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("案例6：NumPy 数组索引与切片")
print("=" * 60)

# ============================================================
# 1. 一维数组索引
# ============================================================
print("\n--- 1. 一维数组索引 ---")

arr1d = np.arange(10, 21)
print(f"一维数组: {arr1d}")
print(f"arr1d[0]   = {arr1d[0]}")
print(f"arr1d[5]   = {arr1d[5]}")
print(f"arr1d[-1]  = {arr1d[-1]}")
print(f"arr1d[-3]  = {arr1d[-3]}")

# 切片：左闭右开
print(f"arr1d[2:6]   = {arr1d[2:6]}")
print(f"arr1d[:5]    = {arr1d[:5]}")
print(f"arr1d[7:]    = {arr1d[7:]}")
print(f"arr1d[:]     = {arr1d[:]}")

# ============================================================
# 2. 二维数组索引
# ============================================================
print("\n--- 2. 二维数组索引 ---")

arr2d = np.arange(1, 13).reshape(3, 4)
print(f"二维数组:\n{arr2d}")
print(f"arr2d[0, 1]    = {arr2d[0, 1]}")
print(f"arr2d[2, 3]    = {arr2d[2, 3]}")
print(f"arr2d[1]       = {arr2d[1]}")
print(f"arr2d[-1, -1]  = {arr2d[-1, -1]}")

# 行切片与列切片
print(f"arr2d[0:2, 1:3] =\n{arr2d[0:2, 1:3]}")
print(f"arr2d[:, 2]     = {arr2d[:, 2]}")
print(f"arr2d[1, :]     = {arr2d[1, :]}")

# ============================================================
# 3. 布尔索引
# ============================================================
print("\n--- 3. 布尔索引 ---")

data = np.array([23, 45, 12, 67, 34, 89, 56, 10])
print(f"原始数据: {data}")

mask = data > 40
print(f"布尔掩码 (data > 40): {mask}")
print(f"布尔索引结果: {data[mask]}")

# 组合条件
mask_and = (data > 20) & (data < 60)
print(f"20 < data < 60: {data[mask_and]}")

mask_or = (data < 15) | (data > 80)
print(f"data < 15 或 data > 80: {data[mask_or]}")

mask_not = ~(data > 40)
print(f"data <= 40: {data[mask_not]}")

# 用布尔索引赋值
data_copy = data.copy()
data_copy[data_copy > 50] = 50
print(f"将 >50 的值截断为 50: {data_copy}")

# ============================================================
# 4. 花式索引（Fancy Indexing）
# ============================================================
print("\n--- 4. 花式索引 ---")

arr = np.arange(0, 100, 10)
print(f"数组: {arr}")

indices = [1, 3, 5, 7]
print(f"arr[[1,3,5,7]] = {arr[indices]}")

# 二维花式索引
arr2d_fancy = np.arange(1, 17).reshape(4, 4)
print(f"\n二维数组:\n{arr2d_fancy}")

rows = [0, 1, 3]
cols = [1, 2, 0]
print(f"arr2d[[0,1,3], [1,2,0]] = {arr2d_fancy[rows, cols]}")

# 使用 np.ix_ 构造索引器
ix_result = arr2d_fancy[np.ix_([0, 2], [1, 3])]
print(f"np.ix_([0,2], [1,3]) 结果:\n{ix_result}")

# ============================================================
# 5. 带步长的切片
# ============================================================
print("\n--- 5. 带步长的切片 ---")

arr_step = np.arange(0, 30)
print(f"数组: {arr_step}")
print(f"arr[::2]   (步长2): {arr_step[::2]}")
print(f"arr[1::3]  (从1开始步长3): {arr_step[1::3]}")
print(f"arr[::-1]  (反转): {arr_step[::-1]}")
print(f"arr[::-2]  (步长-2): {arr_step[::-2]}")
print(f"arr[5:20:4] (5到20步长4): {arr_step[5:20:4]}")

# 二维带步长
arr2d_step = np.arange(1, 26).reshape(5, 5)
print(f"\n二维数组:\n{arr2d_step}")
print(f"每隔一行一列取值:\n{arr2d_step[::2, ::2]}")

# ============================================================
# 6. 负数索引
# ============================================================
print("\n--- 6. 负数索引 ---")

arr_neg = np.array([10, 20, 30, 40, 50, 60, 70])
print(f"数组: {arr_neg}")
print(f"arr[-1]    = {arr_neg[-1]}")
print(f"arr[-3]    = {arr_neg[-3]}")
print(f"arr[-4:-1] = {arr_neg[-4:-1]}")
print(f"arr[-2:]   = {arr_neg[-2:]}")
print(f"arr[:-3]   = {arr_neg[:-3]}")

# 二维负数索引
arr2d_neg = np.arange(1, 10).reshape(3, 3)
print(f"\n二维数组:\n{arr2d_neg}")
print(f"arr2d[-1, -1]  = {arr2d_neg[-1, -1]}")
print(f"arr2d[-2:, -2:] =\n{arr2d_neg[-2:, -2:]}")

# ============================================================
# 7. np.where() 条件索引
# ============================================================
print("\n--- 7. np.where() 条件索引 ---")

scores = np.array([55, 82, 91, 43, 76, 68, 95, 37])
print(f"成绩数据: {scores}")

# np.where 返回满足条件的索引
pass_indices = np.where(scores >= 60)
print(f"及格的索引: {pass_indices[0]}")
print(f"及格的成绩: {scores[pass_indices]}")

# np.where 三元用法：条件为真取 x，否则取 y
result = np.where(scores >= 60, "及格", "不及格")
print(f"及格判定: {result}")

# 多条件 np.where
grade = np.where(scores >= 90, "优秀",
        np.where(scores >= 60, "及格", "不及格"))
print(f"等级划分: {grade}")

# ============================================================
# 8. np.take() 按索引取值
# ============================================================
print("\n--- 8. np.take() 按索引取值 ---")

arr_take = np.array([100, 200, 300, 400, 500, 600])
print(f"数组: {arr_take}")

print(f"np.take(arr, [0, 2, 4])     = {np.take(arr_take, [0, 2, 4])}")
print(f"np.take(arr, [-1, -2])      = {np.take(arr_take, [-1, -2])}")

# 沿指定轴取值
arr2d_take = np.arange(1, 13).reshape(3, 4)
print(f"\n二维数组:\n{arr2d_take}")
print(f"沿 axis=0 取行 [0, 2]:\n{np.take(arr2d_take, [0, 2], axis=0)}")
print(f"沿 axis=1 取列 [1, 3]:\n{np.take(arr2d_take, [1, 3], axis=1)}")

print("\n" + "=" * 60)
print("案例6 演示完毕")
print("=" * 60)
