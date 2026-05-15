from pathlib import Path

import numpy as np

# 数据来源: iris_sepal_length.csv (Iris花萼长度数据集)

# 读取iris数据集中的花萼长度数据（iris_sepal_length.csv），并对其进行统计分析，包括排序、去重，并求出和、累积和、均值、标准差、方差、最小值、最大值。

# 读取文件
iris_sepal_length = np.loadtxt(Path(__file__).parent / 'iris_sepal_length.csv', delimiter=",") 

print('排序前的花萼长度表为：\n',iris_sepal_length)

# 排序
iris_sepal_length.sort()  

print('排序后的花萼长度表为：\n',iris_sepal_length)

# 去除重复值
print('去重后的花萼长度表为：\n',np.unique(iris_sepal_length))

# 计算数组总和
print('花萼长度表的总和为：\n',np.sum(iris_sepal_length)) 

# 计算所有元素的累计和
print('花萼长度表的累计和为：\n',np.cumsum(iris_sepal_length))

# 计算数组均值
print('花萼长度表的均值为：\n',np.mean(iris_sepal_length)) 

# 计算数组标准差
print('花萼长度表的标准差为：\n',np.std(iris_sepal_length))

# 计算数组方差
print('花萼长度表的方差为：\n',np.var(iris_sepal_length))

# 计算最小值
print('花萼长度表的最小值为：\n',np.min(iris_sepal_length))  

# 计算最大值
print('花萼长度表的最大值为：\n',np.max(iris_sepal_length))


