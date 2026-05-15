"""
案例14：3D绘图与等高线图
数据来源：自建数据（numpy生成）
知识点：Axes3D、projection='3d'、plot_surface、scatter3D、plot_wireframe、contour、contourf、colorbar
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(16, 14))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Z值')
ax1.set_title('3D曲面图（plot_surface）')
ax1.set_xlabel('X轴')
ax1.set_ylabel('Y轴')
ax1.set_zlabel('Z轴')

np.random.seed(42)
n = 100
xs = np.random.uniform(-5, 5, n)
ys = np.random.uniform(-5, 5, n)
zs = np.sin(np.sqrt(xs**2 + ys**2)) + np.random.normal(0, 0.1, n)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
sc = ax2.scatter3D(xs, ys, zs, c=zs, cmap='plasma', s=30, alpha=0.8, edgecolor='k', linewidth=0.3)
fig.colorbar(sc, ax=ax2, shrink=0.5, aspect=10, label='Z值')
ax2.set_title('3D散点图（scatter3D）')
ax2.set_xlabel('X轴')
ax2.set_ylabel('Y轴')
ax2.set_zlabel('Z轴')

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.plot_wireframe(X, Y, Z, color='#4E79A7', linewidth=0.5, rstride=5, cstride=5)
ax3.set_title('3D线框图（plot_wireframe）')
ax3.set_xlabel('X轴')
ax3.set_ylabel('Y轴')
ax3.set_zlabel('Z轴')

ax4 = fig.add_subplot(2, 3, 4)
cs = ax4.contour(X, Y, Z, levels=15, colors='black', linewidths=0.8)
ax4.clabel(cs, inline=True, fontsize=8)
ax4.set_title('等高线图（contour）')
ax4.set_xlabel('X轴')
ax4.set_ylabel('Y轴')

ax5 = fig.add_subplot(2, 3, 5)
cf = ax5.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
fig.colorbar(cf, ax=ax5, label='Z值')
ax5.set_title('填充等高线图（contourf）')
ax5.set_xlabel('X轴')
ax5.set_ylabel('Y轴')

Z2 = X * np.exp(-X**2 - Y**2)
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.plot_surface(X, Y, Z2, cmap='coolwarm', alpha=0.85, linewidth=0, antialiased=True)
ax6.set_title('高斯函数3D曲面')
ax6.set_xlabel('X轴')
ax6.set_ylabel('Y轴')
ax6.set_zlabel('Z轴')
ax6.view_init(elev=30, azim=45)

plt.tight_layout()
save_path = output_dir / '14_3D绘图与等高线图.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'图表已保存至: {save_path}')
plt.show()

print('\n=== 3D绘图与等高线图要点 ===')
print('1. fig.add_subplot(projection="3d") 创建3D坐标轴')
print('2. ax.plot_surface(X, Y, Z, cmap) 绘制3D曲面，X/Y/Z为二维网格')
print('3. ax.scatter3D(x, y, z, c, cmap) 绘制3D散点图')
print('4. ax.plot_wireframe(X, Y, Z) 绘制线框图，rstride/cstride控制稀疏度')
print('5. plt.contour(X, Y, Z, levels) 绘制等高线，clabel添加数值标签')
print('6. plt.contourf(X, Y, Z, levels, cmap) 绘制填充等高线')
print('7. fig.colorbar() 添加颜色条，shrink/aspect控制尺寸')
print('8. ax.view_init(elev, azim) 调整3D视角')
