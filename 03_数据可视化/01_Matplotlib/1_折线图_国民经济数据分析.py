# 依赖库最低版本要求: numpy>=1.24, matplotlib>=3.7
# 数据来源: 国民经济核算季度数据.npz
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 绘制 2000-2017年各产业、行业嫉妒生产增加总值发展趋势 “ 折线图 ”
# 加载数据
fp = np.load(Path(__file__).parent / '国民经济核算季度数据.npz', allow_pickle=True)

# 遍历键，获得数组名
for k in fp:
    print(k)

# 获取保存的数组
columns = fp['columns']
values = fp['values']
print('columns :\n', columns)
print('values :\n', values)

###########  绘制图形 ##########

# 1、创建画布
fig = plt.figure(figsize=(8,8))  # 返回画布对象
# fig = plt.figure(figsiez=(50,50),dpi=80)
# 默认不支持中文，需要进行修改，修改完后，不支持符号
plt.rcParams['font.sans-serif'] = ['SimHei']
# 增加字体之后变得不支持符号，需要吸怪RC参数让其他继续支持符号
plt.rcParams['axes.unicode_minus'] = False
# 设置子图间距
plt.subplots_adjust(hspace=0.5)
# wspace = None， 子图之间的宽度间距 ------设置为（0,1]小数 ---子图宽度占比
# hspace = None， 子图之间的高度间距 ------设置为（0,1]小数 ---子图高度占比

# 2、绘制图形和修饰

# 创建子图 1
fig.add_subplot(2, 1, 1)    # 横轴--时间--序号

x = np.arange(values.shape[0])  # TODO

# 纵轴--各个产业的增加总值 生成一个一维数组
# y1 = values[:, 3]
# y2 = values[:, 4]
# y3 = values[:, 5]
y = values[:, 3:6]
# 绘图
# 注意：此时使用一个横轴，对应队列数据，拿出每一类与横轴一一对应
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
plt.plot(x, y)

# 增加标题
plt.title('2000-2017年各个产业、行业增加总值')

# 纵轴名称
plt.ylabel('生成总值（亿元）')

# 从colums中获取
legend = [tmp[:4] for tmp in columns[3:6]]  # TODO

# 创建图例
plt.legend(legend, loc=2, fontsize=5)  # fontsize=5 设置图例大小

# 修改横轴刻度
xticks = values[:, 1]  # 获取时间 如‘2000年第一季度’
plt.xticks(x[::4], xticks[::4], rotation=45)  # rotation 旋转角度 因为太密了设置步长来过度一些 x，轴必须对应

# 创建子图 2
fig.add_subplot(2, 1, 2)

# 横坐标--时间年份
x = np.arange(values.shape[0])
# 纵坐标 y
y = values[:, 6:]
# 绘图
plt.plot(x, y)
# 创建图例
legend = [tmp[:2] for tmp in columns[6:]]
plt.legend(legend, fontsize=5)

# 修改x轴，将序号替换为中文
xticks = values[:, 1]
plt.xticks(x[::4], xticks[::4],rotation=45)

# 3、图形展示
# 保存图片
plt.savefig('./产业行业增加总值_折线图.png')
plt.show()

# 折线图的应用场景
# 用于查看一数据的发展规律、趋势---折线图

