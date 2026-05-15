"""
案例7：NumPy 数组形状变换与展平
==================================
本案例演示 NumPy 数组的形状变换操作，包括：
1. reshape() 重塑形状
2. ravel() 展平（返回视图）
3. flatten() 展平（返回副本）
4. ravel vs flatten 的区别（视图 vs 副本）
5. transpose() 与 T 属性转置
6. swapaxes() 交换轴
7. expand_dims() 扩展维度
8. squeeze() 去除长度为1的维度
9. newaxis 增加维度

数据来源：自行构造的示例数据
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("案例7：NumPy 数组形状变换与展平")
print("=" * 60)

# ============================================================
# 1. reshape() 重塑形状
# ============================================================
print("\n--- 1. reshape() 重塑形状 ---")

arr = np.arange(1, 13)
print(f"原始数组: {arr}")
print(f"原始形状: {arr.shape}")

reshaped = arr.reshape(3, 4)
print(f"reshape(3,4):\n{reshaped}")
print(f"形状: {reshaped.shape}")

reshaped2 = arr.reshape(2, 6)
print(f"reshape(2,6):\n{reshaped2}")

# 使用 -1 自动推断维度
reshaped_auto = arr.reshape(3, -1)
print(f"reshape(3,-1) 自动推断:\n{reshaped_auto}")

reshaped_auto2 = arr.reshape(-1, 4)
print(f"reshape(-1,4) 自动推断:\n{reshaped_auto2}")

# 三维 reshape
reshaped_3d = arr.reshape(2, 2, 3)
print(f"reshape(2,2,3) 三维:\n{reshaped_3d}")

# reshape 返回的是视图（共享内存）
print(f"\nreshape 返回视图验证:")
a = np.arange(6)
b = a.reshape(2, 3)
b[0, 0] = 999
print(f"修改 b 后 a = {a}  (a 也被修改，说明是视图)")

# ============================================================
# 2. ravel() 展平（返回视图）
# ============================================================
print("\n--- 2. ravel() 展平 ---")

arr2d = np.arange(1, 10).reshape(3, 3)
print(f"二维数组:\n{arr2d}")

raveled = arr2d.ravel()
print(f"ravel() 结果: {raveled}")
print(f"ravel 返回类型: {type(raveled)}")

# ravel 默认按行展平（C 顺序）
raveled_c = arr2d.ravel(order='C')
print(f"ravel(order='C') 按行: {raveled_c}")

# 按列展平（Fortran 顺序）
raveled_f = arr2d.ravel(order='F')
print(f"ravel(order='F') 按列: {raveled_f}")

# ============================================================
# 3. flatten() 展平（返回副本）
# ============================================================
print("\n--- 3. flatten() 展平 ---")

arr2d_b = np.arange(1, 10).reshape(3, 3)
print(f"二维数组:\n{arr2d_b}")

flattened = arr2d_b.flatten()
print(f"flatten() 结果: {flattened}")

flattened_c = arr2d_b.flatten(order='C')
print(f"flatten(order='C') 按行: {flattened_c}")

flattened_f = arr2d_b.flatten(order='F')
print(f"flatten(order='F') 按列: {flattened_f}")

# ============================================================
# 4. ravel vs flatten 的区别（视图 vs 副本）
# ============================================================
print("\n--- 4. ravel vs flatten 区别 ---")

original = np.arange(1, 10).reshape(3, 3)
print(f"原始数组:\n{original}")

# ravel 返回视图，修改会影响原数组
ravel_view = original.ravel()
ravel_view[0] = 999
print(f"修改 ravel 结果后，原数组[0,0] = {original[0, 0]}  (被修改，ravel 是视图)")

# 恢复
original[0, 0] = 1

# flatten 返回副本，修改不影响原数组
flatten_copy = original.flatten()
flatten_copy[0] = 888
print(f"修改 flatten 结果后，原数组[0,0] = {original[0, 0]}  (未修改，flatten 是副本)")

# 用 base 属性验证
arr_test = np.arange(6).reshape(2, 3)
r = arr_test.ravel()
f = arr_test.flatten()
print(f"ravel().base is 原数组: {r.base is arr_test}")
print(f"flatten().base is 原数组: {f.base is arr_test}")

# ============================================================
# 5. transpose() 与 T 属性转置
# ============================================================
print("\n--- 5. transpose() 与 T 属性 ---")

arr_t = np.arange(1, 7).reshape(2, 3)
print(f"原始数组 (2x3):\n{arr_t}")

print(f"T 属性转置 (3x2):\n{arr_t.T}")
print(f"transpose() 转置:\n{arr_t.transpose()}")

# 三维数组转置
arr_3d = np.arange(1, 25).reshape(2, 3, 4)
print(f"\n三维数组形状: {arr_3d.shape}")

transposed_3d = arr_3d.transpose(1, 0, 2)
print(f"transpose(1,0,2) 形状: {transposed_3d.shape}")

transposed_3d_2 = arr_3d.transpose(2, 1, 0)
print(f"transpose(2,1,0) 形状: {transposed_3d_2.shape}")

# transpose 返回视图
a_t = np.arange(6).reshape(2, 3)
b_t = a_t.T
print(f"\ntranspose 返回视图验证:")
b_t[0, 0] = 999
print(f"修改转置后，原数组[0,0] = {a_t[0, 0]}  (被修改，是视图)")

# ============================================================
# 6. swapaxes() 交换轴
# ============================================================
print("\n--- 6. swapaxes() 交换轴 ---")

arr_swap = np.arange(1, 13).reshape(3, 4)
print(f"原始数组 (3x4):\n{arr_swap}")

swapped = arr_swap.swapaxes(0, 1)
print(f"swapaxes(0,1) 后 (4x3):\n{swapped}")

# 三维数组交换轴
arr_3d_swap = np.arange(1, 25).reshape(2, 3, 4)
print(f"\n三维数组形状: {arr_3d_swap.shape}")
swapped_3d = arr_3d_swap.swapaxes(0, 2)
print(f"swapaxes(0,2) 后形状: {swapped_3d.shape}")

# swapaxes 也返回视图
a_s = np.arange(6).reshape(2, 3)
b_s = a_s.swapaxes(0, 1)
b_s[0, 0] = 777
print(f"\nswapaxes 返回视图验证: 原数组[0,0] = {a_s[0, 0]}")

# ============================================================
# 7. expand_dims() 扩展维度
# ============================================================
print("\n--- 7. expand_dims() 扩展维度 ---")

arr_exp = np.array([1, 2, 3, 4])
print(f"一维数组: {arr_exp}, 形状: {arr_exp.shape}")

# 在轴0位置插入维度
expanded_0 = np.expand_dims(arr_exp, axis=0)
print(f"expand_dims(axis=0): 形状 {expanded_0.shape}, 值: {expanded_0}")

# 在轴1位置插入维度
expanded_1 = np.expand_dims(arr_exp, axis=1)
print(f"expand_dims(axis=1): 形状 {expanded_1.shape}, 值=\n{expanded_1}")

# 使用元组同时扩展多个维度
expanded_multi = np.expand_dims(arr_exp, axis=(0, 2))
print(f"expand_dims(axis=(0,2)): 形状 {expanded_multi.shape}")

# 二维数组扩展维度
arr2d_exp = np.arange(1, 7).reshape(2, 3)
print(f"\n二维数组形状: {arr2d_exp.shape}")
expanded_2d = np.expand_dims(arr2d_exp, axis=2)
print(f"expand_dims(axis=2): 形状 {expanded_2d.shape}")

# ============================================================
# 8. squeeze() 去除长度为1的维度
# ============================================================
print("\n--- 8. squeeze() 去除长度为1的维度 ---")

arr_sq = np.arange(1, 7).reshape(1, 2, 3, 1)
print(f"原始形状: {arr_sq.shape}")

squeezed_all = np.squeeze(arr_sq)
print(f"squeeze() 全部去除: 形状 {squeezed_all.shape}")

squeezed_axis0 = np.squeeze(arr_sq, axis=0)
print(f"squeeze(axis=0): 形状 {squeezed_axis0.shape}")

squeezed_axis3 = np.squeeze(arr_sq, axis=3)
print(f"squeeze(axis=3): 形状 {squeezed_axis3.shape}")

# squeeze 不能去除长度不为1的维度
try:
    np.squeeze(arr_sq, axis=1)
except ValueError as e:
    print(f"squeeze(axis=1) 报错: {e}")

# ============================================================
# 9. newaxis 增加维度
# ============================================================
print("\n--- 9. newaxis 增加维度 ---")

arr_new = np.array([10, 20, 30, 40])
print(f"一维数组: {arr_new}, 形状: {arr_new.shape}")

# np.newaxis 等价于 None
row_vec = arr_new[np.newaxis, :]
print(f"行向量 [newaxis, :]: 形状 {row_vec.shape}, 值: {row_vec}")

col_vec = arr_new[:, np.newaxis]
print(f"列向量 [:, newaxis]: 形状 {col_vec.shape}, 值=\n{col_vec}")

# 使用 None 等价写法
row_vec_none = arr_new[None, :]
print(f"行向量 [None, :]: 形状 {row_vec_none.shape}")

# 多次使用 newaxis
multi_new = arr_new[np.newaxis, :, np.newaxis]
print(f"[newaxis, :, newaxis]: 形状 {multi_new.shape}")

# newaxis 实际应用：数组运算中的维度对齐
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
outer = a[:, np.newaxis] * b[np.newaxis, :]
print(f"\n外积运算 (3x1 * 1x4):\n{outer}")
print(f"外积结果形状: {outer.shape}")

print("\n" + "=" * 60)
print("案例7 演示完毕")
print("=" * 60)
