# 导入numpy库，用于进行科学计算
import numpy as np

# 初始化第一个矩阵，使用字符串定义矩阵元素，分号分隔行
matri_1 = np.mat("2 3;4 5")
# 打印矩阵matri_1
print(matri_1)

print()

# 初始化第二个矩阵，同样使用字符串定义矩阵元素
matri_2 = np.mat("6 7;8 9")
# 打印矩阵matri_2
print(matri_2)

print()

# 计算并打印两个矩阵的乘积
# 这里使用了numpy的mat对象，它支持矩阵的直接乘法，不同于numpy的array对象
print(matri_1*matri_2)

