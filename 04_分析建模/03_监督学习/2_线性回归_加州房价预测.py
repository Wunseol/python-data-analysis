# 数据来源: sklearn.datasets.fetch_california_housing (加州房价数据集)
# [已弃用] load_boston 在 sklearn 1.2 中已移除，请使用 fetch_california_housing 替代

"""
线性回归 — 加州房价预测
========================
本案例演示:
1. LinearRegression 线性回归模型
2. fit / predict / score 方法
3. 回归评估指标: MSE, MAE, R²
4. 回归系数的解读
5. 预测值 vs 真实值散点图
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、数据加载与划分
# ============================================================

california = fetch_california_housing()
X = california.data
y = california.target

print("=" * 60)
print("【加州房价数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"特征名称: {california.feature_names}")
print(f"目标值 (房价中位数, 单位: 十万美元) 范围: [{y.min():.2f}, {y.max():.2f}]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ============================================================
# 二、构建与训练线性回归模型
# ============================================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【模型参数】")
print(f"截距 (intercept): {model.intercept_:.4f}")
print(f"回归系数 (coefficients):")
for name, coef in zip(california.feature_names, model.coef_):
    print(f"  {name:20s}: {coef:.4f}")

# ============================================================
# 三、模型预测与评估
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 使用 score 方法 (返回 R² 分数)
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

# 使用 sklearn.metrics 计算各项指标
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2_metric = r2_score(y_train, y_train_pred)
test_r2_metric = r2_score(y_test, y_test_pred)

print("\n" + "=" * 60)
print("【模型评估指标】")
print(f"{'指标':<10} {'训练集':>12} {'测试集':>12}")
print("-" * 36)
print(f"{'MSE':<10} {train_mse:>12.4f} {test_mse:>12.4f}")
print(f"{'MAE':<10} {train_mae:>12.4f} {test_mae:>12.4f}")
print(f"{'R²':<10} {train_r2:>12.4f} {test_r2:>12.4f}")

# ============================================================
# 四、回归系数解读
# ============================================================

print("\n" + "=" * 60)
print("【回归系数解读】")
print("正系数: 特征增大时, 房价中位数增大")
print("负系数: 特征增大时, 房价中位数减小")
print("系数绝对值: 反映特征对房价的影响程度 (需考虑特征量纲)")
sorted_indices = np.argsort(np.abs(model.coef_))[::-1]
for idx in sorted_indices:
    name = california.feature_names[idx]
    coef = model.coef_[idx]
    direction = "↑" if coef > 0 else "↓"
    print(f"  {name:20s}: {coef:+.4f} {direction}")

# ============================================================
# 五、预测值 vs 真实值散点图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 训练集散点图
axes[0].scatter(y_train, y_train_pred, alpha=0.3, s=10, color='steelblue')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
             'r--', linewidth=2, label='理想预测线')
axes[0].set_xlabel('真实值 (十万美元)', fontsize=12)
axes[0].set_ylabel('预测值 (十万美元)', fontsize=12)
axes[0].set_title(f'训练集: 预测值 vs 真实值\nR²={train_r2:.4f}', fontsize=13)
axes[0].legend(fontsize=11)

# 测试集散点图
axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10, color='darkorange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='理想预测线')
axes[1].set_xlabel('真实值 (十万美元)', fontsize=12)
axes[1].set_ylabel('预测值 (十万美元)', fontsize=12)
axes[1].set_title(f'测试集: 预测值 vs 真实值\nR²={test_r2:.4f}', fontsize=13)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('线性回归_预测vs真实.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 六、残差分析
# ============================================================

residuals = y_test - y_test_pred
print("\n" + "=" * 60)
print("【残差分析】")
print(f"残差均值: {residuals.mean():.6f}")
print(f"残差标准差: {residuals.std():.4f}")
print(f"残差范围: [{residuals.min():.4f}, {residuals.max():.4f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_pred, residuals, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('预测值', fontsize=12)
axes[0].set_ylabel('残差', fontsize=12)
axes[0].set_title('残差散点图', fontsize=13)

axes[1].hist(residuals, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
axes[1].set_xlabel('残差', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title('残差分布直方图', fontsize=13)

plt.tight_layout()
plt.savefig('线性回归_残差分析.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("线性回归 — 加州房价预测演示完成!")
