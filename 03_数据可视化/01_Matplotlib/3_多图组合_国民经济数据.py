# 数据来源: 国民经济核算季度数据.npz
from pathlib import Path
# 导入绘图库
import matplotlib.pyplot as plt
# 导入科学计算库
import numpy as np

# 加载数据，注意数据文件的路径和格式
fp = np.load(Path(__file__).parent / '国民经济核算季度数据.npz', allow_pickle=True)

# 遍历数据文件中的所有键，以便检查加载的数据内容
for k in fp:
    print(k)
# 遍历一个数据文件中的所有键，而输出表示这些键的数据内容包括columns和values两部分。

print('-------------------------------分割-------------------------------')


# 获取保存的数组
# 从文件指针fp中提取名为'columns'和'values'的数组
columns = fp['columns']
values = fp['values']
# 显示列数组和值数组的内容
print('columns :\n', columns)
print('values :\n', values)

print('-------------------------------分割-------------------------------')

# 创建画布
fig = plt.figure(figsize=(8,8),dpi=80)  # 返回画布对象 #该函数创建一个尺寸为50x50英寸、分辨率为80 dpi的图像对象fig
# 默认不支持中文，需要进行修改，修改完后，不支持符号，该行代码设置matplotlib库中显示中文所需的字体为“黑体”，使得绘图时能正常显示中文标签。
plt.rcParams['font.sans-serif'] = ['SimHei']
# 增加字体之后变得不支持符号，需要吸怪RC参数让其他继续支持符号
plt.rcParams['axes.unicode_minus'] = False
# 设置子图间距
plt.subplots_adjust(hspace=0.5)
# wspace = None， 子图之间的宽度间距 ------设置为（0,1]小数 ---子图宽度占比
# hspace = None， 子图之间的高度间距 ------设置为（0,1]小数 ---子图高度占比


# 绘制图形和修饰

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


plt.title('2000-2017年各个产业、行业增加总值')  # 增加标题
plt.ylabel('生成总值（亿元）')  # 纵轴名称

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



plt.figure(figsize=(10, 10))
plt.scatter(values[:, 0], values[:, 2], marker='o')  # 画散点图
plt.xlabel('年份')
plt.ylabel('生产总值（亿元）')
plt.ylim((0, 225000))
plt.xticks(range(0, 70, 4), values[range(0, 70, 4), 1], rotation=45)  # 此时取得值都是第一季度的
plt.title('2000-2017年季度生产总值散点图')
plt.savefig('./2000-2017年季度生产总值散点图.png')  # 图片要先保存再显示
plt.show()  # 在当前设备显示图片





plt.figure(figsize=(6,5))## 设置画布
plt.bar(range(3),values[-1,3:6],width = 0.5)## 绘制散点图
plt.xlabel('产业')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加y轴名称
label = ['第一产业','第二产业','第三产业']## 刻度标签
plt.xticks(range(3),label)
plt.title('2017年第一季度各产业国民生产总值直方图')## 添加图表标题
plt.savefig('./2017年第一季度各产业国民生产总值直方图.png')
plt.show()





plt.figure(figsize=(6,6))## 将画布设定为正方形，则绘制的饼图是正圆
label= ['第一产业','第二产业','第三产业']## 定义饼状图的标签，标签是列表
explode = [0.01,0.01,0.01]## 设定各项离心n个半径
plt.pie(values[-1,3:6],explode=explode,labels=label,
        autopct='%1.1f%%')## 绘制饼图
plt.title('2017年第一季度各产业国民生产总值饼图')
plt.savefig('./2017年第一季度各产业生产总值占比饼图')
plt.show()



# 该Python脚本从指定文件加载国民经济数据，并使用matplotlib库绘制多种图表。

# 加载数据：从.npz文件加载数据，并打印键名和具体数据。
# 配置绘图环境：
# 设置画布大小。
# 配置字体以支持中文显示。
# 调整子图间距。
# 绘制折线图：
# 创建两个子图，分别绘制不同产业的增加值随时间的变化。
# 设置图例、标题、坐标轴标签等。
# 绘制散点图：展示2000年至2017年每个季度的生产总值。
# 绘制柱状图：显示2017年第一季度各产业的国民生产总值。
# 绘制饼图：展示2017年第一季度各产业生产总值的占比情况。
# 所有图表均保存为PNG文件并显示在屏幕上。


