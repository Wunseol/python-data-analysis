"""
案例9：NumPy 广播机制与通用函数
==================================
本案例演示 NumPy 的广播机制与通用函数（ufunc），包括：
1. 广播规则详解
2. np.broadcast() 广播对象
3. 算术通用函数（+, -, *, /）
4. 比较通用函数
5. np.add() 加法
6. np.multiply() 乘法
7. np.sqrt() 平方根
8. np.exp() 指数
9. np.log() 对数
10. np.maximum() 逐元素最大值
11. np.mod() 取模

数据来源：自行构造的示例数据
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("案例9：NumPy 广播机制与通用函数")
print("=" * 60)

# ============================================================
# 1. 广播规则详解
# ============================================================
print("\n--- 1. 广播规则详解 ---")
print("广播三规则:")
print("  1. 如果两个数组维度数不同，小维度数组的形状在左边补1")
print("  2. 如果两个数组在某个维度上大小不同，大小为1的维度会被拉伸")
print("  3. 如果两个数组在某个维度上大小不同且都不为1，则报错")

# 标量与数组
arr_scalar = np.array([1, 2, 3, 4])
result_scalar = arr_scalar + 10
print(f"\n标量广播: [1,2,3,4] + 10 = {result_scalar}")

# 一维与二维
arr_2d = np.ones((3, 4))
arr_1d = np.array([1, 2, 3, 4])
result_2d_1d = arr_2d + arr_1d
print(f"\n二维(3,4) + 一维(4,):\n{result_2d_1d}")

# 列向量与行向量
col = np.array([[1], [2], [3]])
row = np.array([10, 20, 30, 40])
print(f"\n列向量形状: {col.shape}")
print(f"行向量形状: {row.shape}")
result_outer = col + row
print(f"列(3,1) + 行(4,) 广播结果 形状{result_outer.shape}:\n{result_outer}")

# 广播失败示例
print("\n广播失败示例:")
try:
    a_fail = np.ones((3, 4))
    b_fail = np.ones((3, 5))
    _ = a_fail + b_fail
except ValueError as e:
    print(f"  (3,4) + (3,5) 报错: {e}")

# ============================================================
# 2. np.broadcast() 广播对象
# ============================================================
print("\n--- 2. np.broadcast() 广播对象 ---")

x = np.array([1, 2, 3])
y = np.array([[10], [20]])
print(f"x 形状: {x.shape}")
print(f"y 形状: {y.shape}")

b = np.broadcast(x, y)
print(f"广播后形状: {b.shape}")
print(f"广播结果（手动迭代）:")
for val in b:
    print(f"  {val}", end="")
print()

# 使用 broadcast_to 显式广播
arr_bt = np.array([1, 2, 3])
broadcasted = np.broadcast_to(arr_bt, (4, 3))
print(f"\nbroadcast_to([1,2,3], (4,3)):\n{broadcasted}")

# broadcast_arrays 同时广播多个数组
a_ba = np.array([[1, 2, 3]])
b_ba = np.array([[10], [20], [30]])
ba_result = np.broadcast_arrays(a_ba, b_ba)
print(f"\nbroadcast_arrays 结果:")
for i, arr in enumerate(ba_result):
    print(f"  数组{i} 形状{arr.shape}:\n{arr}")

# ============================================================
# 3. 算术通用函数（+, -, *, /）
# ============================================================
print("\n--- 3. 算术通用函数 ---")

a_arith = np.array([10, 20, 30, 40])
b_arith = np.array([1, 2, 3, 4])

print(f"a = {a_arith}")
print(f"b = {b_arith}")
print(f"a + b = {a_arith + b_arith}")
print(f"a - b = {a_arith - b_arith}")
print(f"a * b = {a_arith * b_arith}")
print(f"a / b = {a_arith / b_arith}")
print(f"a // b = {a_arith // b_arith}")
print(f"a ** b = {a_arith ** b_arith}")
print(f"a % b = {a_arith % b_arith}")

# 对应的 ufunc 写法
print(f"\n对应的 ufunc 写法:")
print(f"np.add(a, b)     = {np.add(a_arith, b_arith)}")
print(f"np.subtract(a, b)= {np.subtract(a_arith, b_arith)}")
print(f"np.multiply(a, b)= {np.multiply(a_arith, b_arith)}")
print(f"np.divide(a, b)  = {np.divide(a_arith, b_arith)}")
print(f"np.floor_divide(a,b) = {np.floor_divide(a_arith, b_arith)}")
print(f"np.power(a, b)   = {np.power(a_arith, b_arith)}")
print(f"np.mod(a, b)     = {np.mod(a_arith, b_arith)}")

# ============================================================
# 4. 比较通用函数
# ============================================================
print("\n--- 4. 比较通用函数 ---")

a_cmp = np.array([3, 7, 2, 9, 5])
b_cmp = np.array([4, 6, 2, 8, 5])

print(f"a = {a_cmp}")
print(f"b = {b_cmp}")
print(f"a == b  = {a_cmp == b_cmp}")
print(f"a != b  = {a_cmp != b_cmp}")
print(f"a > b   = {a_cmp > b_cmp}")
print(f"a < b   = {a_cmp < b_cmp}")
print(f"a >= b  = {a_cmp >= b_cmp}")
print(f"a <= b  = {a_cmp <= b_cmp}")

# 对应的 ufunc 写法
print(f"\n对应的 ufunc 写法:")
print(f"np.equal(a, b)       = {np.equal(a_cmp, b_cmp)}")
print(f"np.not_equal(a, b)   = {np.not_equal(a_cmp, b_cmp)}")
print(f"np.greater(a, b)     = {np.greater(a_cmp, b_cmp)}")
print(f"np.less(a, b)        = {np.less(a_cmp, b_cmp)}")
print(f"np.greater_equal(a,b)= {np.greater_equal(a_cmp, b_cmp)}")
print(f"np.less_equal(a, b)  = {np.less_equal(a_cmp, b_cmp)}")

# ============================================================
# 5. np.add() 加法
# ============================================================
print("\n--- 5. np.add() 加法 ---")

a_add = np.array([1, 2, 3])
b_add = np.array([10, 20, 30])

print(f"np.add(a, b) = {np.add(a_add, b_add)}")

# 指定 out 参数（原地运算）
out_arr = np.zeros(3)
np.add(a_add, b_add, out=out_arr)
print(f"使用 out 参数: {out_arr}")

# where 参数（条件运算）
result_where = np.add(a_add, b_add, where=[True, False, True])
print(f"add(where=[T,F,T]): {result_where}")

# ============================================================
# 6. np.multiply() 乘法
# ============================================================
print("\n--- 6. np.multiply() 乘法 ---")

a_mul = np.array([2, 3, 4])
b_mul = np.array([5, 6, 7])

print(f"np.multiply(a, b) = {np.multiply(a_mul, b_mul)}")

# 标量乘法
print(f"np.multiply(a, 10) = {np.multiply(a_mul, 10)}")

# 矩阵逐元素乘法（非矩阵乘法）
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
print(f"\n逐元素乘法:\n{np.multiply(m1, m2)}")
print(f"矩阵乘法 (@):\n{m1 @ m2}")

# ============================================================
# 7. np.sqrt() 平方根
# ============================================================
print("\n--- 7. np.sqrt() 平方根 ---")

arr_sqrt = np.array([1, 4, 9, 16, 25, 36])
print(f"原数组: {arr_sqrt}")
print(f"np.sqrt(): {np.sqrt(arr_sqrt)}")

# 浮点数
arr_sqrt_f = np.array([2.0, 3.0, 5.0])
print(f"np.sqrt([2,3,5]): {np.sqrt(arr_sqrt_f)}")

# 负数产生 nan
arr_neg = np.array([-1, 4, -9])
print(f"np.sqrt([-1,4,-9]): {np.sqrt(arr_neg)}")

# ============================================================
# 8. np.exp() 指数
# ============================================================
print("\n--- 8. np.exp() 指数 ---")

arr_exp = np.array([0, 1, 2, 3])
print(f"原数组: {arr_exp}")
print(f"np.exp(): {np.exp(arr_exp)}")
print(f"验证 e^0 = {np.exp(0)}, e^1 = {np.exp(1):.6f}")

# exp2 以2为底
print(f"np.exp2([0,1,2,3]): {np.exp2(arr_exp)}")

# ============================================================
# 9. np.log() 对数
# ============================================================
print("\n--- 9. np.log() 对数 ---")

arr_log = np.array([1, np.e, np.e**2, np.e**3])
print(f"原数组: {arr_log}")
print(f"np.log() 自然对数: {np.log(arr_log)}")

# log2 以2为底
arr_log2 = np.array([1, 2, 4, 8, 16])
print(f"np.log2([1,2,4,8,16]): {np.log2(arr_log2)}")

# log10 以10为底
arr_log10 = np.array([1, 10, 100, 1000])
print(f"np.log10([1,10,100,1000]): {np.log10(arr_log10)}")

# log1p 计算 log(1+x)，对 x 接近0时更精确
small_x = np.array([1e-15, 1e-10, 1e-5])
print(f"\nlog(1+x) 精度对比:")
print(f"  np.log(1+x): {np.log(1 + small_x)}")
print(f"  np.log1p(x): {np.log1p(small_x)}")

# ============================================================
# 10. np.maximum() 逐元素最大值
# ============================================================
print("\n--- 10. np.maximum() 逐元素最大值 ---")

a_max = np.array([3, 7, 2, 9, 1])
b_max = np.array([5, 4, 8, 6, 3])

print(f"a = {a_max}")
print(f"b = {b_max}")
print(f"np.maximum(a, b) = {np.maximum(a_max, b_max)}")

# 标量比较
print(f"np.maximum(a, 5) = {np.maximum(a_max, 5)}")

# 与 np.max 的区别
print(f"\nnp.maximum vs np.max 的区别:")
print(f"  np.maximum(a, b) 逐元素比较: {np.maximum(a_max, b_max)}")
print(f"  np.max(a) 求整个数组最大值: {np.max(a_max)}")

# 处理 nan
a_nan = np.array([3, np.nan, 2])
b_nan = np.array([5, 4, np.nan])
print(f"\nnp.maximum([3,nan,2], [5,4,nan]) = {np.maximum(a_nan, b_nan)}")
print(f"np.fmax([3,nan,2], [5,4,nan])    = {np.fmax(a_nan, b_nan)}")

# ============================================================
# 11. np.mod() 取模
# ============================================================
print("\n--- 11. np.mod() 取模 ---")

a_mod = np.array([10, 17, 23, 30, 7])
b_mod = np.array([3, 5, 7, 4, 2])

print(f"a = {a_mod}")
print(f"b = {b_mod}")
print(f"np.mod(a, b) = {np.mod(a_mod, b_mod)}")
print(f"a % b        = {a_mod % b_mod}")

# 标量取模
print(f"np.mod(a, 5) = {np.mod(a_mod, 5)}")

# np.remainder 等价于 np.mod
print(f"np.remainder(a, b) = {np.remainder(a_mod, b_mod)}")

# np.fmod 处理负数的方式不同（C语言风格）
neg_a = np.array([-7, -7, 7, 7])
neg_b = np.array([3, -3, 3, -3])
print(f"\n负数取模对比:")
print(f"  a = {neg_a}")
print(f"  b = {neg_b}")
print(f"  np.mod(a, b)  = {np.mod(neg_a, neg_b)}")
print(f"  np.fmod(a, b) = {np.fmod(neg_a, neg_b)}")

print("\n" + "=" * 60)
print("案例9 演示完毕")
print("=" * 60)
