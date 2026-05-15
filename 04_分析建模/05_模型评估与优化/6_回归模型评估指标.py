# 数据来源: sklearn.datasets.fetch_california_housing (加州房价数据集)

"""
回归模型评估指标
================
本案例演示:
1. MAE 平均绝对误差
2. MSE 均方误差
3. RMSE 均方根误差
4. R² 决定系数
5. adjusted R² 调整决定系数
6. MAPE 平均绝对百分比误差
7. explained_variance_score 解释方差分数
8. 残差分析
9. QQ 图检验残差正态性
10. 交叉验证回归指标
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    make_scorer
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、数据加载与准备
# ============================================================

housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

print("=" * 60)
print("【加州房价数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"目标变量: 房价中位数 (单位: 十万美元)")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"目标变量均值: {y.mean():.2f}, 标准差: {y.std():.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# ============================================================
# 二、模型训练
# ============================================================

models = {
    '线性回归': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearRegression())
    ]),
    'Ridge回归': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', Ridge(alpha=1.0))
    ]),
    'Lasso回归': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', Lasso(alpha=0.01, max_iter=5000))
    ]),
    '随机森林': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

# ============================================================
# 三、MAE 平均绝对误差
# ============================================================

print("\n" + "=" * 60)
print("【MAE 平均绝对误差】")
print("MAE = (1/n) * Σ|y_true - y_pred|")
print("含义: 预测值与真实值的平均绝对偏差")
print("优点: 直观, 对异常值不敏感 (与MSE相比)")
print("")

for name, y_pred in predictions.items():
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  {name:<12}: MAE = {mae:.4f}")

# ============================================================
# 四、MSE 均方误差
# ============================================================

print("\n" + "=" * 60)
print("【MSE 均方误差】")
print("MSE = (1/n) * Σ(y_true - y_pred)²")
print("含义: 预测值与真实值的平均平方偏差")
print("优点: 对大误差惩罚更重, 数学性质好 (可微)")
print("缺点: 量纲与原数据不同 (平方), 对异常值敏感")
print("")

for name, y_pred in predictions.items():
    mse = mean_squared_error(y_test, y_pred)
    print(f"  {name:<12}: MSE = {mse:.4f}")

# ============================================================
# 五、RMSE 均方根误差
# ============================================================

print("\n" + "=" * 60)
print("【RMSE 均方根误差】")
print("RMSE = √MSE = √[(1/n) * Σ(y_true - y_pred)²]")
print("含义: MSE 的平方根, 量纲与原数据一致")
print("RMSE >= MAE 恒成立, 差异越大说明误差分布越不均匀")
print("")

for name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  {name:<12}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, RMSE/MAE = {rmse/mae:.4f}")

# ============================================================
# 六、R² 决定系数
# ============================================================

print("\n" + "=" * 60)
print("【R² 决定系数】")
print("R² = 1 - Σ(y_true - y_pred)² / Σ(y_true - y_mean)²")
print("含义: 模型解释的方差占总方差的比例")
print("  R² = 1.0: 完美拟合")
print("  R² = 0.0: 等同于预测均值")
print("  R² < 0.0: 比预测均值还差")
print("")

for name, y_pred in predictions.items():
    r2 = r2_score(y_test, y_pred)
    print(f"  {name:<12}: R² = {r2:.4f}")

# ============================================================
# 七、adjusted R² 调整决定系数
# ============================================================

print("\n" + "=" * 60)
print("【adjusted R² 调整决定系数】")
print("adjusted R² = 1 - [(1-R²)(n-1) / (n-p-1)]")
print("含义: 对 R² 进行自由度修正, 防止增加无用特征导致 R² 虚高")
print("n = 样本数, p = 特征数")
print("")

n = len(y_test)
p = X_test.shape[1]

for name, y_pred in predictions.items():
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"  {name:<12}: R² = {r2:.4f}, adjusted R² = {adj_r2:.4f}, 差值 = {r2 - adj_r2:.6f}")

print(f"\n样本数 n={n}, 特征数 p={p}")
print("当特征数远小于样本数时, R² 和 adjusted R² 差异很小")
print("当特征数接近样本数时, adjusted R² 会明显低于 R²")

# ============================================================
# 八、MAPE 平均绝对百分比误差
# ============================================================

print("\n" + "=" * 60)
print("【MAPE 平均绝对百分比误差】")
print("MAPE = (1/n) * Σ|y_true - y_pred| / |y_true|")
print("含义: 预测误差相对于真实值的百分比")
print("优点: 无量纲, 可跨数据集比较")
print("注意: y_true 接近0时, MAPE 会趋于无穷大")
print("")

for name, y_pred in predictions.items():
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"  {name:<12}: MAPE = {mape:.4f} ({mape*100:.2f}%)")

# ============================================================
# 九、explained_variance_score 解释方差分数
# ============================================================

print("\n" + "=" * 60)
print("【explained_variance_score 解释方差分数】")
print("EVS = 1 - Var(y_true - y_pred) / Var(y_true)")
print("含义: 模型解释的方差比例")
print("与 R² 的区别: R² 使用残差平方和, EVS 使用残差方差")
print("当模型无系统偏差时, EVS ≈ R²")
print("")

for name, y_pred in predictions.items():
    evs = explained_variance_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  {name:<12}: EVS = {evs:.4f}, R² = {r2:.4f}, 差值 = {evs - r2:.6f}")

# ============================================================
# 十、残差分析
# ============================================================

y_pred_best = predictions['随机森林']
residuals = y_test - y_pred_best

print("\n" + "=" * 60)
print("【残差分析 (随机森林)】")
print(f"残差均值: {residuals.mean():.6f} (接近0表示无系统偏差)")
print(f"残差标准差: {residuals.std():.4f}")
print(f"残差偏度: {stats.skew(residuals):.4f} (接近0表示对称)")
print(f"残差峰度: {stats.kurtosis(residuals):.4f} (接近0表示正态)")

print("\n残差分析要点:")
print("  1. 残差应随机分布在0附近, 无明显模式")
print("  2. 残差 vs 预测值图: 不应有漏斗形状 (异方差性)")
print("  3. 残差直方图: 应近似正态分布")
print("  4. QQ图: 点应沿对角线分布")

# ============================================================
# 十一、QQ 图检验残差正态性
# ============================================================

print("\n" + "=" * 60)
print("【QQ 图检验残差正态性】")

shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])
print(f"Shapiro-Wilk 正态性检验 (前5000个样本):")
print(f"  统计量: {shapiro_stat:.6f}")
print(f"  p值: {shapiro_p:.6f}")
if shapiro_p > 0.05:
    print(f"  结论: p={shapiro_p:.4f} > 0.05, 不能拒绝正态性假设")
else:
    print(f"  结论: p={shapiro_p:.4f} < 0.05, 拒绝正态性假设")
    print(f"  注意: 大样本下 Shapiro-Wilk 检验非常敏感, 即使轻微偏离也会拒绝")
    print(f"        建议结合 QQ 图和直方图综合判断")

# ============================================================
# 十二、交叉验证回归指标
# ============================================================

print("\n" + "=" * 60)
print("【交叉验证回归指标】")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring_methods = {
    'neg_MAE': 'neg_mean_absolute_error',
    'neg_MSE': 'neg_mean_squared_error',
    'r2': 'r2',
    'neg_MAPE': 'neg_mean_absolute_percentage_error'
}

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    for score_name, scoring in scoring_methods.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        if score_name.startswith('neg_'):
            display_scores = -scores
            print(f"  {score_name.replace('neg_', ''):<8}: "
                  f"均值={display_scores.mean():.4f}, 标准差={display_scores.std():.4f}")
        else:
            print(f"  {score_name:<8}: "
                  f"均值={scores.mean():.4f}, 标准差={scores.std():.4f}")

# ============================================================
# 十三、综合指标对比表
# ============================================================

print("\n" + "=" * 60)
print("【综合指标对比表】")
print(f"{'模型':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'adj_R²':>8} {'MAPE':>8} {'EVS':>8}")
print("-" * 64)

for name, y_pred in predictions.items():
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print(f"{name:<12} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {adj_r2:>8.4f} {mape:>8.4f} {evs:>8.4f}")

# ============================================================
# 十四、可视化
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. 预测值 vs 真实值 (随机森林)
ax = axes[0, 0]
ax.scatter(y_test, y_pred_best, alpha=0.3, s=10, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='完美预测线')
ax.set_xlabel('真实值', fontsize=12)
ax.set_ylabel('预测值', fontsize=12)
ax.set_title('预测值 vs 真实值 (随机森林)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 2. 残差 vs 预测值
ax = axes[0, 1]
ax.scatter(y_pred_best, residuals, alpha=0.3, s=10, color='steelblue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('预测值', fontsize=12)
ax.set_ylabel('残差', fontsize=12)
ax.set_title('残差 vs 预测值 (随机森林)', fontsize=13)
ax.grid(True, alpha=0.3)

# 3. 残差直方图
ax = axes[0, 2]
ax.hist(residuals, bins=50, density=True, color='steelblue', alpha=0.7, edgecolor='white')
x_range = np.linspace(residuals.min(), residuals.max(), 200)
pdf = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
ax.plot(x_range, pdf, 'r-', linewidth=2, label='正态分布拟合')
ax.set_xlabel('残差', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('残差分布直方图', fontsize=13)
ax.legend(fontsize=10)

# 4. QQ 图
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('QQ 图 (残差正态性检验)', fontsize=13)
ax.grid(True, alpha=0.3)

# 5. 模型 R² 对比
ax = axes[1, 1]
model_names = list(predictions.keys())
r2_values = [r2_score(y_test, predictions[name]) for name in model_names]
rmse_values = [np.sqrt(mean_squared_error(y_test, predictions[name])) for name in model_names]

bars = ax.bar(model_names, r2_values, color=['steelblue', 'darkorange', 'green', 'purple'], alpha=0.8)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('各模型 R² 对比', fontsize=13)
for bar, r2 in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{r2:.4f}', ha='center', fontsize=11)
ax.set_ylim(0, max(r2_values) + 0.1)
ax.tick_params(axis='x', rotation=15)

# 6. 各模型 RMSE 对比
ax = axes[1, 2]
bars2 = ax.bar(model_names, rmse_values, color=['steelblue', 'darkorange', 'green', 'purple'], alpha=0.8)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('各模型 RMSE 对比', fontsize=13)
for bar, rmse in zip(bars2, rmse_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{rmse:.4f}', ha='center', fontsize=11)
ax.set_ylim(0, max(rmse_values) + 0.1)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '回归模型评估指标.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("回归模型评估指标演示完成!")
