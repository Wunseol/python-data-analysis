"""
案例13：箱线图与误差图
数据来源：自建数据（numpy生成）
知识点：plt.boxplot()、plt.violinplot()、plt.errorbar()、箱线图组成（中位数/Q1/Q3/须/离群点）、分布比较
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

np.random.seed(42)
data_a = np.random.normal(70, 10, 200)
data_b = np.random.normal(75, 15, 200)
data_c = np.random.normal(65, 8, 200)
data_d = np.random.normal(80, 12, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax1 = axes[0, 0]
bp = ax1.boxplot([data_a, data_b, data_c, data_d],
                 labels=['班级A', '班级B', '班级C', '班级D'],
                 patch_artist=True,
                 boxprops=dict(linewidth=1.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 medianprops=dict(linewidth=2, color='red'),
                 flierprops=dict(marker='o', markersize=5, markerfacecolor='red'))
box_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title('箱线图：四班成绩分布比较')
ax1.set_ylabel('分数')
ax1.grid(axis='y', alpha=0.3)

print('=== 班级A箱线图统计 ===')
print(f'  中位数: {np.median(data_a):.2f}')
print(f'  Q1（25%）: {np.percentile(data_a, 25):.2f}')
print(f'  Q3（75%）: {np.percentile(data_a, 75):.2f}')
print(f'  IQR: {np.percentile(data_a, 75) - np.percentile(data_a, 25):.2f}')
print(f'  最小值: {np.min(data_a):.2f}')
print(f'  最大值: {np.max(data_a):.2f}')

ax2 = axes[0, 1]
vp = ax2.violinplot([data_a, data_b, data_c, data_d],
                    showmeans=True, showmedians=True, showextrema=True)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(box_colors[i])
    body.set_alpha(0.7)
vp['cmeans'].set_color('blue')
vp['cmedians'].set_color('red')
vp['cmeans'].set_linewidth(2)
vp['cmedians'].set_linewidth(2)
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['班级A', '班级B', '班级C', '班级D'])
ax2.set_title('小提琴图：分布密度与统计量')
ax2.set_ylabel('分数')
ax2.grid(axis='y', alpha=0.3)

ax3 = axes[1, 0]
methods = ['方法1', '方法2', '方法3', '方法4', '方法5']
means = np.array([72.5, 78.3, 68.9, 82.1, 75.6])
std_devs = np.array([5.2, 8.1, 3.5, 6.7, 4.9])
x_pos = np.arange(len(methods))
ax3.bar(x_pos, means, color=box_colors[:2] + ['#59A14F', '#EDC948', '#B07AA1'],
        edgecolor='black', linewidth=0.8, alpha=0.7, width=0.5)
ax3.errorbar(x_pos, means, yerr=std_devs, fmt='none', ecolor='black',
             capsize=6, capthick=1.5, linewidth=1.5)
for i, (m, s) in enumerate(zip(means, std_devs)):
    ax3.text(i, m + s + 1.5, f'{m:.1f}±{s:.1f}', ha='center', fontsize=9)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods)
ax3.set_title('柱状图 + 误差棒（标准差）')
ax3.set_ylabel('得分')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)

ax4 = axes[1, 1]
x_line = np.arange(1, 11)
y_line = np.array([23, 25, 28, 30, 35, 33, 38, 40, 42, 45])
y_err = np.array([2.1, 1.8, 3.0, 2.5, 3.5, 2.8, 3.2, 2.0, 2.7, 3.0])
ax4.errorbar(x_line, y_line, yerr=y_err, fmt='o-', color='#4E79A7',
             ecolor='#E15759', capsize=5, capthick=1.5,
             linewidth=2, markersize=6, markerfacecolor='white',
             markeredgecolor='#4E79A7', markeredgewidth=1.5,
             label='测量值 ± 误差')
ax4.fill_between(x_line, y_line - y_err, y_line + y_err,
                 alpha=0.15, color='#4E79A7', label='误差范围')
ax4.set_title('折线误差图（plt.errorbar）')
ax4.set_xlabel('测量点')
ax4.set_ylabel('测量值')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
save_path = output_dir / '13_箱线图与误差图.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n图表已保存至: {save_path}')
plt.show()

print('\n=== 箱线图与误差图要点 ===')
print('1. 箱线图组成：中位数线 → Q1~Q3箱体 → 1.5*IQR须 → 离群点')
print('2. plt.boxplot(patch_artist=True) 启用填充色')
print('3. 小提琴图结合了箱线图和核密度估计，展示分布形状')
print('4. plt.errorbar(x, y, yerr, fmt, capsize) 绘制误差棒')
print('5. 柱状图 + errorbar 组合可直观展示均值与离散程度')
