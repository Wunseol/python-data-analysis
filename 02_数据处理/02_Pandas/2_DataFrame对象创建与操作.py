# 数据来源: 自建示例数据
# 本脚本演示 Pandas DataFrame 对象的创建与基本操作

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、DataFrame 的创建")
print("=" * 60)

# 1. 从字典创建 DataFrame
data_dict = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 28, 35],
    '城市': ['北京', '上海', '广州', '深圳'],
    '薪资': [15000, 22000, 18000, 25000]
}
df_from_dict = pd.DataFrame(data_dict)
print("\n从字典创建 DataFrame:")
print(df_from_dict)

# 2. 从列表创建 DataFrame
data_list = [
    ['张三', 25, '北京', 15000],
    ['李四', 30, '上海', 22000],
    ['王五', 28, '广州', 18000],
    ['赵六', 35, '深圳', 25000]
]
df_from_list = pd.DataFrame(data_list, columns=['姓名', '年龄', '城市', '薪资'])
print("\n从列表创建 DataFrame:")
print(df_from_list)

# 3. 从 NumPy 数组创建 DataFrame
arr = np.random.randint(60, 100, size=(4, 3))
df_from_ndarray = pd.DataFrame(arr, columns=['语文', '数学', '英语'], index=['学生A', '学生B', '学生C', '学生D'])
print("\n从 NumPy 数组创建 DataFrame:")
print(df_from_ndarray)

# 4. 从 Series 字典创建
s1 = pd.Series([85, 90, 78], index=['期中', '期末', '平时'])
s2 = pd.Series([88, 76, 92], index=['期中', '期末', '平时'])
df_from_series = pd.DataFrame({'语文': s1, '数学': s2})
print("\n从 Series 字典创建 DataFrame:")
print(df_from_series)

print("\n" + "=" * 60)
print("二、columns 和 index 参数")
print("=" * 60)

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df_custom = pd.DataFrame(data, index=['row1', 'row2', 'row3'])
print("\n自定义行索引:")
print(df_custom)

# 选择部分列并重排列顺序
df_select = pd.DataFrame(data, columns=['C', 'A'], index=['x', 'y', 'z'])
print("\n选择部分列并自定义索引:")
print(df_select)

print("\n" + "=" * 60)
print("三、DataFrame 基本属性")
print("=" * 60)

df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 28, 35, 22],
    '城市': ['北京', '上海', '广州', '深圳', '杭州'],
    '薪资': [15000, 22000, 18000, 25000, 12000]
})

print(f"\n列名 (columns): {df.columns.tolist()}")
print(f"行索引 (index): {df.index.tolist()}")
print(f"数据类型 (dtypes):\n{df.dtypes}")
print(f"形状 (shape): {df.shape}")
print(f"元素总数 (size): {df.size}")
print(f"维度 (ndim): {df.ndim}")

print("\n" + "=" * 60)
print("四、head() 和 tail()")
print("=" * 60)

print("\nhead() 默认前5行:")
print(df.head())
print("\nhead(2) 前2行:")
print(df.head(2))
print("\ntail(2) 后2行:")
print(df.tail(2))

print("\n" + "=" * 60)
print("五、info() 和 describe()")
print("=" * 60)

print("\ninfo():")
df.info()

print("\ndescribe():")
print(df.describe())

print("\ndescribe(include='all'):")
print(df.describe(include='all'))

print("\n" + "=" * 60)
print("六、转置 T")
print("=" * 60)

print("\n转置后的 DataFrame:")
print(df.T)

print("\n" + "=" * 60)
print("七、values 属性")
print("=" * 60)

print(f"\nvalues 类型: {type(df.values)}")
print(f"values 形状: {df.values.shape}")
print(f"values 内容:\n{df.values}")

print("\n" + "=" * 60)
print("八、列的增删改")
print("=" * 60)

df_ops = df.copy()

# 增加列
df_ops['部门'] = ['技术部', '市场部', '技术部', '财务部', '市场部']
print("\n增加 '部门' 列:")
print(df_ops)

# 修改列
df_ops['薪资'] = df_ops['薪资'] + 1000
print("\n薪资增加1000:")
print(df_ops)

# 删除列
df_dropped = df_ops.drop(columns=['部门'])
print("\n删除 '部门' 列:")
print(df_dropped)

# 使用 del 删除
df_ops_del = df_ops.copy()
del df_ops_del['部门']
print("\n使用 del 删除 '部门' 列:")
print(df_ops_del)

print("\n" + "=" * 60)
print("九、行索引重置")
print("=" * 60)

df_reset = df.set_index('姓名')
print("\n设置 '姓名' 为索引:")
print(df_reset)

df_restore = df_reset.reset_index()
print("\n重置索引:")
print(df_restore)

df_reset_drop = df_reset.reset_index(drop=True)
print("\n重置索引并丢弃原索引:")
print(df_reset_drop)
