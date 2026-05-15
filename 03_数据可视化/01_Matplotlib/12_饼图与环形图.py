"""
案例12：饼图与环形图
数据来源：自建数据（numpy生成）
知识点：plt.pie()、labels、autopct、startangle、explode、colors、shadow、环形图wedgeprops、百分比格式化
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

labels = ['食品', '交通', '住房', '娱乐', '教育', '其他']
sizes = np.array([35, 15, 25, 10, 10, 5])
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.set_title('基础饼图')

ax2 = axes[0, 1]
explode = (0.05, 0, 0, 0, 0, 0.1)
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, shadow=True,
        textprops={'fontsize': 11})
ax2.set_title('饼图：explode突出 + shadow阴影')

ax3 = axes[1, 0]
ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45,
        colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
ax3.set_title('环形图（wedgeprops width=0.5）')

ax4 = axes[1, 1]
inner_sizes = np.array([20, 15, 40, 25])
inner_labels = ['必需品', '非必需品', '投资', '储蓄']
inner_colors = ['#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
outer_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']

ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=outer_colors, radius=1.0,
        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=1.5))
ax4.pie(inner_sizes, labels=inner_labels, autopct='%1.1f%%', startangle=90,
        colors=inner_colors, radius=0.65,
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5))
ax4.set_title('双层环形图（嵌套饼图）')

plt.tight_layout()
save_path = output_dir / '12_饼图与环形图.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'图表已保存至: {save_path}')
plt.show()

print('\n=== 饼图与环形图要点 ===')
print('1. plt.pie(sizes, labels, autopct, startangle, colors) 基本参数')
print('2. autopct 百分比格式：%1.1f%% 保留1位小数，%d%% 整数')
print('3. explode 参数让某块突出，值越大偏移越远')
print('4. shadow=True 添加阴影效果增强立体感')
print('5. 环形图：wedgeprops=dict(width=0.5) 设置环的宽度')
print('6. 双层环形图：嵌套调用两次 pie()，外层 radius 大，内层 radius 小')
