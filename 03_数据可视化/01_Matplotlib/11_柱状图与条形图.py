"""
案例11：柱状图与条形图
数据来源：自建数据（numpy生成）
知识点：plt.bar()、plt.barh()、分组柱状图、堆叠柱状图、自定义颜色、edgecolor、width、柱上数值标注
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

categories = ['第一季度', '第二季度', '第三季度', '第四季度']
sales_a = np.array([120, 150, 170, 200])
sales_b = np.array([100, 130, 160, 180])
sales_c = np.array([80, 110, 140, 160])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
bars = ax1.bar(categories, sales_a, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
for bar, val in zip(bars, sales_a):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_title('基础柱状图（plt.bar）')
ax1.set_ylabel('销售额（万元）')
ax1.set_ylim(0, 230)
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[0, 1]
bars_h = ax2.barh(categories, sales_a, color=colors, edgecolor='black', linewidth=0.8, height=0.5)
for bar, val in zip(bars_h, sales_a):
    ax2.text(val + 3, bar.get_y() + bar.get_height() / 2,
             str(val), ha='left', va='center', fontsize=11, fontweight='bold')
ax2.set_title('水平条形图（plt.barh）')
ax2.set_xlabel('销售额（万元）')
ax2.set_xlim(0, 240)
ax2.grid(axis='x', alpha=0.3)

ax3 = axes[1, 0]
x = np.arange(len(categories))
bar_width = 0.25
bars1 = ax3.bar(x - bar_width, sales_a, bar_width, label='产品A',
                color='#4E79A7', edgecolor='black', linewidth=0.8)
bars2 = ax3.bar(x, sales_b, bar_width, label='产品B',
                color='#F28E2B', edgecolor='black', linewidth=0.8)
bars3 = ax3.bar(x + bar_width, sales_c, bar_width, label='产品C',
                color='#E15759', edgecolor='black', linewidth=0.8)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
ax3.set_title('分组柱状图')
ax3.set_ylabel('销售额（万元）')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.set_ylim(0, 230)
ax3.grid(axis='y', alpha=0.3)

ax4 = axes[1, 1]
bars_bottom = ax4.bar(categories, sales_a, label='产品A',
                      color='#4E79A7', edgecolor='black', linewidth=0.8, width=0.5)
bars_middle = ax4.bar(categories, sales_b, bottom=sales_a, label='产品B',
                      color='#F28E2B', edgecolor='black', linewidth=0.8, width=0.5)
bars_top = ax4.bar(categories, sales_c, bottom=sales_a + sales_b, label='产品C',
                   color='#E15759', edgecolor='black', linewidth=0.8, width=0.5)
for i, (a, b, c) in enumerate(zip(sales_a, sales_b, sales_c)):
    total = a + b + c
    ax4.text(i, total + 5, str(total), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_title('堆叠柱状图')
ax4.set_ylabel('销售额（万元）')
ax4.legend()
ax4.set_ylim(0, 620)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = output_dir / '11_柱状图与条形图.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'图表已保存至: {save_path}')
plt.show()

print('\n=== 柱状图与条形图要点 ===')
print('1. plt.bar(x, height, width, color, edgecolor) 绘制垂直柱状图')
print('2. plt.barh(y, width, height, color) 绘制水平条形图')
print('3. 分组柱状图：通过偏移 x 坐标 + 设置 width 实现并列')
print('4. 堆叠柱状图：使用 bottom 参数指定堆叠起始位置')
print('5. 柱上数值标注：bar.get_x() + bar.get_width()/2 定位中心')
