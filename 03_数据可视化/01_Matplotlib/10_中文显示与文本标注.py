"""
案例10：中文显示与文本标注
数据来源：自建数据（numpy生成）
知识点：rcParams字体设置、plt.text()、plt.annotate()箭头标注、fontproperties参数
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(x, y_sin, color='blue', linewidth=2)
ax1.set_xlabel('横轴（弧度）')
ax1.set_ylabel('纵轴（幅值）')
ax1.set_title('基本中文标题：正弦曲线')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(x, y_sin, color='blue', linewidth=2)
ax2.plot(x, y_cos, color='red', linewidth=2)
ax2.text(np.pi / 2, 1.05, '正弦峰值', ha='center', fontsize=12, color='blue')
ax2.text(np.pi, -0.95, '余弦谷值', ha='center', fontsize=12, color='red')
ax2.text(3.5, 0.5, '这是plt.text()\n添加的文本标注', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax2.set_title('plt.text() 文本标注')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(x, y_sin, color='blue', linewidth=2)
ax3.annotate('最大值点',
             xy=(np.pi / 2, 1),
             xytext=(np.pi / 2 + 1, 0.5),
             fontsize=12,
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             color='red')
ax3.annotate('零点',
             xy=(np.pi, 0),
             xytext=(np.pi + 0.8, -0.5),
             fontsize=12,
             arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
             color='green')
ax3.annotate('起点',
             xy=(0, 0),
             xytext=(0.5, -0.7),
             fontsize=12,
             arrowprops=dict(arrowstyle='fancy', color='purple', lw=2),
             color='purple')
ax3.set_title('plt.annotate() 箭头标注')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=14)
ax4.plot(x, y_sin, color='blue', linewidth=2)
ax4.set_title('fontproperties 参数示例', fontproperties=font_prop)
ax4.set_xlabel('横轴', fontproperties=font_prop)
ax4.set_ylabel('纵轴', fontproperties=font_prop)
ax4.text(np.pi, 0.5, '自定义字体', fontproperties=font_prop, color='red')
ax4.annotate('标注文字',
             xy=(np.pi / 2, 1),
             xytext=(2.5, 0.3),
             fontproperties=font_prop,
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax4.grid(True, alpha=0.3)

plt.tight_layout()
save_path = output_dir / '10_中文显示与文本标注.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'图表已保存至: {save_path}')
plt.show()

print('\n=== 中文显示与文本标注要点 ===')
print('1. rcParams["font.sans-serif"] 设置中文字体（SimHei/微软雅黑等）')
print('2. rcParams["axes.unicode_minus"] = False 解决负号显示问题')
print('3. plt.text(x, y, s) 在指定坐标添加文本')
print('4. plt.annotate() 通过 xy 和 xytext 配合 arrowprops 实现箭头标注')
print('5. fontproperties 参数可单独为某个文本指定字体')
