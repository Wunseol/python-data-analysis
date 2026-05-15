from pathlib import Path

import numpy as np

# 数据来源: iris_sepal_length.csv (Iris花萼长度数据集)

# 读取iris数据集中的花萼长度数据，并且对其进行排序、去重、求和、累积和、均值标准差、方差、最大值、最小值。
iris = np.loadtxt(Path(__file__).parent / 'iris_sepal_length.csv')

print('原始数据为:\n', iris)

#排序后的数据
iris.sort()  # 对数据进行排序
print('花萼长度表为:\n', iris)

# 去除重复值
print('去重后的花萼长度表为:\n', np.unique(iris))

# 均值
print('花萼长度均值为:\n', np.mean(iris))

# 计算数据的标准差、方差
print('方差:%f ----标准差:%f\n' % (np.var(iris), np.std(iris)))

# 最大值最小值
print('最大值%f ----最小值:%f\n' % (np.max(iris), np.min(iris)))

