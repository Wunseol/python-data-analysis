"""
案例8：NumPy 数组组合与分割
==============================
本案例演示 NumPy 数组的组合与分割操作，包括：
1. np.concatenate() 通用拼接
2. np.vstack() 垂直堆叠
3. np.hstack() 水平堆叠
4. np.dstack() 深度堆叠
5. np.split() 等分分割
6. np.vsplit() 垂直分割
7. np.hsplit() 水平分割
8. np.array_split() 不等分分割

数据来源：自行构造的示例数据
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("案例8：NumPy 数组组合与分割")
print("=" * 60)

# ============================================================
# 1. np.concatenate() 通用拼接
# ============================================================
print("\n--- 1. np.concatenate() 通用拼接 ---")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 一维拼接（默认 axis=0）
result_1d = np.concatenate([a, b])
print(f"一维拼接: {result_1d}")

# 二维拼接
c = np.arange(1, 7).reshape(2, 3)
d = np.arange(7, 13).reshape(2, 3)
print(f"\n数组 c:\n{c}")
print(f"数组 d:\n{d}")

# 沿轴0拼接（垂直方向，行增加）
result_axis0 = np.concatenate([c, d], axis=0)
print(f"concatenate(axis=0) 垂直拼接:\n{result_axis0}")

# 沿轴1拼接（水平方向，列增加）
result_axis1 = np.concatenate([c, d], axis=1)
print(f"concatenate(axis=1) 水平拼接:\n{result_axis1}")

# 拼接多个数组
e = np.array([[100, 200, 300]])
result_multi = np.concatenate([c, d, e], axis=0)
print(f"拼接三个数组:\n{result_multi}")

# ============================================================
# 2. np.vstack() 垂直堆叠
# ============================================================
print("\n--- 2. np.vstack() 垂直堆叠 ---")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

vstack_1d = np.vstack([v1, v2])
print(f"一维数组 vstack:\n{vstack_1d}")
print(f"形状: {vstack_1d.shape}")

v3 = np.arange(1, 7).reshape(2, 3)
v4 = np.arange(7, 13).reshape(2, 3)
vstack_2d = np.vstack([v3, v4])
print(f"二维数组 vstack:\n{vstack_2d}")
print(f"形状: {vstack_2d.shape}")

# vstack 等价于 concatenate(axis=0)
vstack_eq = np.concatenate([v3, v4], axis=0)
print(f"vstack 与 concatenate(axis=0) 结果一致: {np.array_equal(vstack_2d, vstack_eq)}")

# ============================================================
# 3. np.hstack() 水平堆叠
# ============================================================
print("\n--- 3. np.hstack() 水平堆叠 ---")

h1 = np.array([1, 2, 3])
h2 = np.array([4, 5, 6])

hstack_1d = np.hstack([h1, h2])
print(f"一维数组 hstack: {hstack_1d}")

h3 = np.arange(1, 7).reshape(2, 3)
h4 = np.arange(7, 13).reshape(2, 3)
hstack_2d = np.hstack([h3, h4])
print(f"二维数组 hstack:\n{hstack_2d}")
print(f"形状: {hstack_2d.shape}")

# hstack 等价于 concatenate(axis=1)
hstack_eq = np.concatenate([h3, h4], axis=1)
print(f"hstack 与 concatenate(axis=1) 结果一致: {np.array_equal(hstack_2d, hstack_eq)}")

# ============================================================
# 4. np.dstack() 深度堆叠
# ============================================================
print("\n--- 4. np.dstack() 深度堆叠 ---")

d1 = np.array([[1, 2], [3, 4]])
d2 = np.array([[5, 6], [7, 8]])
d3 = np.array([[9, 10], [11, 12]])

print(f"d1:\n{d1}")
print(f"d2:\n{d2}")

dstack_result = np.dstack([d1, d2])
print(f"dstack 结果:\n{dstack_result}")
print(f"形状: {dstack_result.shape}")

dstack_3 = np.dstack([d1, d2, d3])
print(f"三个数组 dstack 形状: {dstack_3.shape}")

# 一维数组 dstack
d1_1d = np.array([1, 2, 3])
d2_1d = np.array([4, 5, 6])
dstack_1d = np.dstack([d1_1d, d2_1d])
print(f"\n一维 dstack 形状: {dstack_1d.shape}, 值: {dstack_1d}")

# ============================================================
# 5. np.split() 等分分割
# ============================================================
print("\n--- 5. np.split() 等分分割 ---")

arr_split = np.arange(1, 13)
print(f"原始数组: {arr_split}")

# 等分为3份
split_3 = np.split(arr_split, 3)
for i, part in enumerate(split_3):
    print(f"  第{i+1}份: {part}")

# 按指定位置分割
split_pos = np.split(arr_split, [3, 7])
for i, part in enumerate(split_pos):
    print(f"  在位置[3,7]分割 第{i+1}份: {part}")

# 二维数组按行分割
arr2d_split = np.arange(1, 13).reshape(4, 3)
print(f"\n二维数组:\n{arr2d_split}")

split_2d = np.split(arr2d_split, 2, axis=0)
for i, part in enumerate(split_2d):
    print(f"  按行分割 第{i+1}份:\n{part}")

# 如果不能等分则报错
try:
    np.split(arr_split, 5)
except ValueError as e:
    print(f"\n不能等分为5份时报错: {e}")

# ============================================================
# 6. np.vsplit() 垂直分割
# ============================================================
print("\n--- 6. np.vsplit() 垂直分割 ---")

arr_vsplit = np.arange(1, 25).reshape(6, 4)
print(f"原始数组 (6x4):\n{arr_vsplit}")

# 等分为2份
vsplit_2 = np.vsplit(arr_vsplit, 2)
for i, part in enumerate(vsplit_2):
    print(f"  vsplit 第{i+1}份 形状{part.shape}:\n{part}")

# 按指定行位置分割
vsplit_pos = np.vsplit(arr_vsplit, [2, 4])
for i, part in enumerate(vsplit_pos):
    print(f"  vsplit在行[2,4] 第{i+1}份 形状{part.shape}:\n{part}")

# vsplit 等价于 split(axis=0)
vsplit_eq = np.split(arr_vsplit, 2, axis=0)
print(f"vsplit 与 split(axis=0) 结果一致: {all(np.array_equal(a, b) for a, b in zip(vsplit_2, vsplit_eq))}")

# ============================================================
# 7. np.hsplit() 水平分割
# ============================================================
print("\n--- 7. np.hsplit() 水平分割 ---")

arr_hsplit = np.arange(1, 25).reshape(4, 6)
print(f"原始数组 (4x6):\n{arr_hsplit}")

# 等分为3份
hsplit_3 = np.hsplit(arr_hsplit, 3)
for i, part in enumerate(hsplit_3):
    print(f"  hsplit 第{i+1}份 形状{part.shape}:\n{part}")

# 按指定列位置分割
hsplit_pos = np.hsplit(arr_hsplit, [2, 4])
for i, part in enumerate(hsplit_pos):
    print(f"  hsplit在列[2,4] 第{i+1}份 形状{part.shape}:\n{part}")

# hsplit 等价于 split(axis=1)
hsplit_eq = np.split(arr_hsplit, 3, axis=1)
print(f"hsplit 与 split(axis=1) 结果一致: {all(np.array_equal(a, b) for a, b in zip(hsplit_3, hsplit_eq))}")

# ============================================================
# 8. np.array_split() 不等分分割
# ============================================================
print("\n--- 8. np.array_split() 不等分分割 ---")

arr_asplit = np.arange(1, 11)
print(f"原始数组 (10个元素): {arr_asplit}")

# 分为3份（不能等分，array_split 会尽量均匀分配）
asplit_3 = np.array_split(arr_asplit, 3)
for i, part in enumerate(asplit_3):
    print(f"  array_split 第{i+1}份: {part}")

# 分为4份
asplit_4 = np.array_split(arr_asplit, 4)
for i, part in enumerate(asplit_4):
    print(f"  array_split 4份 第{i+1}份: {part}")

# 对比 split 不能不等分
print(f"\nnp.split 无法处理不等分，而 array_split 可以")

# 二维数组不等分
arr2d_asplit = np.arange(1, 13).reshape(4, 3)
print(f"\n二维数组 (4x3):\n{arr2d_asplit}")

asplit_2d = np.array_split(arr2d_asplit, 3, axis=0)
for i, part in enumerate(asplit_2d):
    print(f"  array_split 3份(按行) 第{i+1}份 形状{part.shape}:\n{part}")

# 按指定位置分割
asplit_pos = np.array_split(arr_asplit, [3, 5, 8])
for i, part in enumerate(asplit_pos):
    print(f"  array_split 在[3,5,8] 第{i+1}份: {part}")

print("\n" + "=" * 60)
print("案例8 演示完毕")
print("=" * 60)
