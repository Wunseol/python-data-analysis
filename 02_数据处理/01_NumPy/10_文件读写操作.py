"""
案例10：NumPy 文件读写操作
=============================
本案例演示 NumPy 的文件读写操作，包括：
1. np.save() 保存单个数组为 .npy 文件
2. np.load() 加载 .npy 文件
3. np.savez() 保存多个数组为 .npz 文件
4. np.savetxt() 保存数组为文本文件
5. np.loadtxt() 从文本文件加载数组
6. np.genfromtxt() 处理缺失数据的文本加载
7. .npy 和 .npz 格式说明
8. 使用 genfromtxt 处理缺失数据

数据来源：自行构造的示例数据
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("案例10：NumPy 文件读写操作")
print("=" * 60)

# 设置工作目录
work_dir = Path(__file__).parent / "temp_io_demo"
work_dir.mkdir(exist_ok=True)
print(f"临时文件目录: {work_dir}")

# ============================================================
# 1. np.save() 保存单个数组为 .npy 文件
# ============================================================
print("\n--- 1. np.save() 保存 .npy 文件 ---")

arr_save = np.arange(1, 13).reshape(3, 4)
print(f"待保存数组:\n{arr_save}")
print(f"数组 dtype: {arr_save.dtype}")

npy_path = work_dir / "array_data.npy"
np.save(npy_path, arr_save)
print(f"已保存到: {npy_path}")
print(f"文件是否存在: {npy_path.exists()}")
print(f"文件大小: {npy_path.stat().st_size} 字节")

# 保存浮点数组
arr_float = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
np.save(work_dir / "float_data.npy", arr_float)
print(f"浮点数组已保存")

# ============================================================
# 2. np.load() 加载 .npy 文件
# ============================================================
print("\n--- 2. np.load() 加载 .npy 文件 ---")

loaded_arr = np.load(npy_path)
print(f"加载的数组:\n{loaded_arr}")
print(f"加载后 dtype: {loaded_arr.dtype}")
print(f"与原数组一致: {np.array_equal(arr_save, loaded_arr)}")

# 加载浮点数组
loaded_float = np.load(work_dir / "float_data.npy")
print(f"加载浮点数组: {loaded_float}")

# allow_pickle 参数（安全提示）
print(f"\n注意: 加载 .npy 文件时如包含对象数组需设置 allow_pickle=True")
print(f"  但出于安全考虑，默认 allow_pickle=False")

# ============================================================
# 3. np.savez() 保存多个数组为 .npz 文件
# ============================================================
print("\n--- 3. np.savez() 保存 .npz 文件 ---")

data_x = np.linspace(0, 2 * np.pi, 5)
data_y = np.sin(data_x)
data_label = np.array(["point1", "point2", "point3", "point4", "point5"])

npz_path = work_dir / "multi_data.npz"
np.savez(npz_path, x=data_x, y=data_y, labels=data_label)
print(f"已保存多个数组到: {npz_path}")
print(f"文件大小: {npz_path.stat().st_size} 字节")

# 加载 .npz 文件
loaded_npz = np.load(npz_path)
print(f"\n加载 .npz 文件:")
print(f"  包含的数组名: {list(loaded_npz.keys())}")
print(f"  x = {loaded_npz['x']}")
print(f"  y = {loaded_npz['y']}")
print(f"  labels = {loaded_npz['labels']}")

# savez_compressed 压缩保存
npz_comp_path = work_dir / "multi_data_compressed.npz"
np.savez_compressed(npz_comp_path, x=data_x, y=data_y, labels=data_label)
print(f"\n压缩保存文件大小: {npz_comp_path.stat().st_size} 字节")
print(f"普通保存文件大小: {npz_path.stat().st_size} 字节")

# 不指定键名时自动命名为 arr_0, arr_1, ...
npz_auto_path = work_dir / "auto_named.npz"
np.savez(npz_auto_path, data_x, data_y)
loaded_auto = np.load(npz_auto_path)
print(f"\n自动命名键: {list(loaded_auto.keys())}")

# ============================================================
# 4. np.savetxt() 保存数组为文本文件
# ============================================================
print("\n--- 4. np.savetxt() 保存文本文件 ---")

arr_txt = np.arange(1, 10).reshape(3, 3)
print(f"待保存数组:\n{arr_txt}")

txt_path = work_dir / "data.txt"
np.savetxt(txt_path, arr_txt, fmt='%d', delimiter=',')
print(f"已保存到: {txt_path}")

# 查看文件内容
print(f"文件内容:")
print(txt_path.read_text())

# 保存浮点数，指定格式
arr_float_txt = np.array([[1.23456, 2.34567], [3.45678, 4.56789]])
float_txt_path = work_dir / "float_data.txt"
np.savetxt(float_txt_path, arr_float_txt, fmt='%.2f', delimiter='\t')
print(f"浮点数据文件内容:")
print(float_txt_path.read_text())

# 添加表头
header_txt_path = work_dir / "data_with_header.txt"
np.savetxt(header_txt_path, arr_txt, fmt='%d', delimiter=',',
           header='col1,col2,col3', comments='')
print(f"带表头文件内容:")
print(header_txt_path.read_text())

# ============================================================
# 5. np.loadtxt() 从文本文件加载数组
# ============================================================
print("\n--- 5. np.loadtxt() 加载文本文件 ---")

# 加载整数数据
loaded_txt = np.loadtxt(txt_path, delimiter=',', dtype=int)
print(f"加载整数数据:\n{loaded_txt}")

# 加载浮点数据
loaded_float_txt = np.loadtxt(float_txt_path, delimiter='\t')
print(f"加载浮点数据:\n{loaded_float_txt}")

# 跳过表头加载
loaded_header = np.loadtxt(header_txt_path, delimiter=',', skiprows=1, dtype=int)
print(f"跳过表头加载:\n{loaded_header}")

# 指定列加载
loaded_cols = np.loadtxt(txt_path, delimiter=',', dtype=int, usecols=(0, 2))
print(f"只加载第0和第2列:\n{loaded_cols}")

# unpack 参数：按列解包
col0, col2 = np.loadtxt(txt_path, delimiter=',', dtype=int, usecols=(0, 2), unpack=True)
print(f"解包第0列: {col0}")
print(f"解包第2列: {col2}")

# ============================================================
# 6. np.genfromtxt() 处理缺失数据的文本加载
# ============================================================
print("\n--- 6. np.genfromtxt() 处理缺失数据 ---")

# 创建包含缺失值的 CSV 文件
csv_path = work_dir / "data_with_missing.csv"
csv_content = """姓名,语文,数学,英语
张三,85,92,78
李四,90,,88
王五,76,85,
赵六,,91,82"""
csv_path.write_text(csv_content, encoding='utf-8')
print(f"含缺失值的CSV内容:\n{csv_content}")

# 使用 genfromtxt 加载
data_gen = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=float,
                         filling_values=-1)
print(f"\ngenfromtxt 加载 (filling_values=-1):\n{data_gen}")

# 默认缺失值填充为 nan
data_gen_nan = np.genfromtxt(csv_path, delimiter=',', skip_header=1, dtype=float)
print(f"genfromtxt 加载 (默认nan):\n{data_gen_nan}")

# ============================================================
# 7. .npy 和 .npz 格式说明
# ============================================================
print("\n--- 7. .npy 和 .npz 格式说明 ---")

print(".npy 格式:")
print("  - NumPy 专用的二进制格式")
print("  - 保存数组的形状、dtype 等元信息")
print("  - 读写速度快，占用空间小")
print("  - 适合保存单个数组")
print("  - 跨平台兼容")

print("\n.npz 格式:")
print("  - NumPy 的多数组打包格式（本质是 zip 压缩包）")
print("  - 可保存多个命名数组")
print("  - savez() 普通保存，savez_compressed() 压缩保存")
print("  - 加载后返回 NpzFile 对象，类似字典访问")

# 格式对比
arr_compare = np.random.rand(1000, 10)

npy_cmp_path = work_dir / "compare.npy"
np.save(npy_cmp_path, arr_compare)

txt_cmp_path = work_dir / "compare.txt"
np.savetxt(txt_cmp_path, arr_compare, fmt='%.8f', delimiter=',')

npz_cmp_path = work_dir / "compare.npz"
np.savez_compressed(npz_cmp_path, data=arr_compare)

print(f"\n1000x10 浮点数组存储对比:")
print(f"  .npy  文件大小: {npy_cmp_path.stat().st_size:>10,} 字节")
print(f"  .txt  文件大小: {txt_cmp_path.stat().st_size:>10,} 字节")
print(f"  .npz  文件大小: {npz_cmp_path.stat().st_size:>10,} 字节 (压缩)")

# ============================================================
# 8. 使用 genfromtxt 处理缺失数据（进阶）
# ============================================================
print("\n--- 8. genfromtxt 处理缺失数据（进阶） ---")

# 创建更复杂的缺失数据文件
advanced_path = work_dir / "advanced_missing.csv"
advanced_content = """1.5,2.3,3.7
4.2,NaN,6.1
,8.9,9.3
10.1,11.2,"""
advanced_path.write_text(advanced_content, encoding='utf-8')
print(f"进阶缺失数据:\n{advanced_content}")

# 加载并指定缺失值处理
data_adv = np.genfromtxt(advanced_path, delimiter=',', filling_values=np.nan)
print(f"\n加载结果:\n{data_adv}")

# 检测缺失值
print(f"\n缺失值检测:")
print(f"np.isnan(data_adv):\n{np.isnan(data_adv)}")
print(f"每行缺失值数量: {np.isnan(data_adv).sum(axis=1)}")
print(f"总缺失值数量: {np.isnan(data_adv).sum()}")

# 用列均值填充缺失值
col_means = np.nanmean(data_adv, axis=0)
print(f"\n各列均值: {col_means}")

data_filled = data_adv.copy()
for col_idx in range(data_filled.shape[1]):
    mask = np.isnan(data_filled[:, col_idx])
    data_filled[mask, col_idx] = col_means[col_idx]
print(f"用列均值填充后:\n{data_filled}")

# genfromtxt 的 names 参数（结构化数组）
struct_path = work_dir / "structured.csv"
struct_content = """Alice,85,92
Bob,90,88
Carol,76,95"""
struct_path.write_text(struct_content, encoding='utf-8')

data_struct = np.genfromtxt(struct_path, delimiter=',', dtype=None,
                            names=['name', 'math', 'english'], encoding='utf-8')
print(f"\n结构化数组加载:")
print(f"  dtype: {data_struct.dtype}")
print(f"  数据: {data_struct}")
print(f"  math 列: {data_struct['math']}")
print(f"  name 列: {data_struct['name']}")

# ============================================================
# 清理临时文件
# ============================================================
print("\n--- 清理临时文件 ---")
import shutil
shutil.rmtree(work_dir)
print(f"已删除临时目录: {work_dir}")

print("\n" + "=" * 60)
print("案例10 演示完毕")
print("=" * 60)
