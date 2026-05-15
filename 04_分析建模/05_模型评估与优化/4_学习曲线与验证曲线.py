# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集), sklearn.datasets.load_wine (红酒数据集)

"""
学习曲线与验证曲线
==================
本案例演示:
1. learning_curve 学习曲线
2. validation_curve 验证曲线
3. train_sizes 参数
4. 过拟合/欠拟合诊断
5. 偏差-方差权衡可视化
6. n_jobs 并行参数
7. 曲线解读方法
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import (
    train_test_split, learning_curve, validation_curve, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、数据加载与准备
# ============================================================

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

wine = load_wine()
X_wine = wine.data
y_wine = wine.target

print("=" * 60)
print("【数据集信息】")
print(f"鸢尾花: 特征={X_iris.shape[1]}, 样本={X_iris.shape[0]}, 类别={len(np.unique(y_iris))}")
print(f"红酒:   特征={X_wine.shape[1]}, 样本={X_wine.shape[0]}, 类别={len(np.unique(y_wine))}")

X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.3, stratify=y_wine, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 二、learning_curve 学习曲线
# ============================================================

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, random_state=42))
])

train_sizes_abs, train_scores, val_scores = learning_curve(
    pipe_lr, X_wine, y_wine,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    n_jobs=-1,
    return_times=False
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

print("\n" + "=" * 60)
print("【learning_curve 学习曲线 (LogisticRegression)】")
print(f"train_sizes: {train_sizes_abs}")
print(f"\n{'训练样本数':<12} {'训练分数':>12} {'验证分数':>12} {'差距':>10}")
print("-" * 48)
for i, size in enumerate(train_sizes_abs):
    gap = train_scores_mean[i] - val_scores_mean[i]
    print(f"{size:<12} {train_scores_mean[i]:>12.4f} {val_scores_mean[i]:>12.4f} {gap:>10.4f}")

# ============================================================
# 三、train_sizes 参数详解
# ============================================================

print("\n" + "=" * 60)
print("【train_sizes 参数详解】")
print("train_sizes 支持两种格式:")
print("  1. 浮点数列表: 如 [0.1, 0.3, 0.5, 0.7, 1.0], 表示训练集比例")
print("  2. 整数列表:   如 [20, 50, 100, 150], 表示绝对样本数")
print("")

train_sizes_ratio = np.linspace(0.1, 1.0, 5)
train_sizes_abs_list = [int(len(X_wine) * r) for r in train_sizes_ratio]

print(f"比例方式: {train_sizes_ratio}")
print(f"对应样本数: {train_sizes_abs_list}")

# ============================================================
# 四、不同模型的学习曲线对比
# ============================================================

models = {
    '逻辑回归': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, random_state=42))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42))
    ]),
    '决策树 (无限制)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    '决策树 (max_depth=3)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=3, random_state=42))
    ])
}

print("\n" + "=" * 60)
print("【不同模型的学习曲线对比】")

learning_data = {}
for name, model in models.items():
    ts, tr_scores, va_scores = learning_curve(
        model, X_wine, y_wine,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv, n_jobs=-1
    )
    tr_mean = np.mean(tr_scores, axis=1)
    va_mean = np.mean(va_scores, axis=1)
    learning_data[name] = {
        'train_sizes': ts,
        'train_mean': tr_mean,
        'val_mean': va_mean,
        'train_std': np.std(tr_scores, axis=1),
        'val_std': np.std(va_scores, axis=1)
    }
    final_gap = tr_mean[-1] - va_mean[-1]
    print(f"  {name:<20}: 验证分数={va_mean[-1]:.4f}, 训练-验证差距={final_gap:.4f}")

# ============================================================
# 五、过拟合/欠拟合诊断
# ============================================================

print("\n" + "=" * 60)
print("【过拟合/欠拟合诊断】")

pipe_overfit = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42))
])

pipe_underfit = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, C=0.001, random_state=42))
])

pipe_good = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=10.0, gamma=0.1, random_state=42))
])

diagnose_models = {
    '过拟合 (决策树无限制)': pipe_overfit,
    '欠拟合 (逻辑回归C=0.001)': pipe_underfit,
    '良好拟合 (SVC调参)': pipe_good
}

diagnose_data = {}
for name, model in diagnose_models.items():
    ts, tr_scores, va_scores = learning_curve(
        model, X_wine, y_wine,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv, n_jobs=-1
    )
    diagnose_data[name] = {
        'train_sizes': ts,
        'train_mean': np.mean(tr_scores, axis=1),
        'val_mean': np.mean(va_scores, axis=1),
        'train_std': np.std(tr_scores, axis=1),
        'val_std': np.std(va_scores, axis=1)
    }
    tr_final = np.mean(tr_scores, axis=1)[-1]
    va_final = np.mean(va_scores, axis=1)[-1]
    gap = tr_final - va_final
    print(f"  {name:<30}: 训练={tr_final:.4f}, 验证={va_final:.4f}, 差距={gap:.4f}")

print("\n诊断规则:")
print("  过拟合: 训练分数高, 验证分数低, 差距大 => 增加正则化/减少模型复杂度/增加数据")
print("  欠拟合: 训练分数低, 验证分数低, 差距小 => 增加模型复杂度/减少正则化/增加特征")
print("  良好拟合: 训练和验证分数都较高, 差距适中")

# ============================================================
# 六、validation_curve 验证曲线
# ============================================================

pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', random_state=42))
])

# C 参数的验证曲线
param_range_C = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
train_scores_C, val_scores_C = validation_curve(
    pipe_svc, X_wine, y_wine,
    param_name='clf__C',
    param_range=param_range_C,
    cv=cv,
    n_jobs=-1
)

print("\n" + "=" * 60)
print("【validation_curve 验证曲线 (SVC C参数)】")
print(f"{'C值':<10} {'训练分数':>12} {'验证分数':>12} {'差距':>10}")
print("-" * 46)
for i, c in enumerate(param_range_C):
    tr_mean = np.mean(train_scores_C[i])
    va_mean = np.mean(val_scores_C[i])
    gap = tr_mean - va_mean
    print(f"{c:<10} {tr_mean:>12.4f} {va_mean:>12.4f} {gap:>10.4f}")

best_C_idx = np.argmax(np.mean(val_scores_C, axis=1))
print(f"\n最佳 C 值: {param_range_C[best_C_idx]}, 验证分数: {np.mean(val_scores_C[best_C_idx]):.4f}")

# gamma 参数的验证曲线
param_range_gamma = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
train_scores_gamma, val_scores_gamma = validation_curve(
    pipe_svc, X_wine, y_wine,
    param_name='clf__gamma',
    param_range=param_range_gamma,
    cv=cv,
    n_jobs=-1
)

print("\n" + "=" * 60)
print("【validation_curve 验证曲线 (SVC gamma参数)】")
print(f"{'gamma值':<10} {'训练分数':>12} {'验证分数':>12} {'差距':>10}")
print("-" * 46)
for i, g in enumerate(param_range_gamma):
    tr_mean = np.mean(train_scores_gamma[i])
    va_mean = np.mean(val_scores_gamma[i])
    gap = tr_mean - va_mean
    print(f"{g:<10} {tr_mean:>12.4f} {va_mean:>12.4f} {gap:>10.4f}")

best_gamma_idx = np.argmax(np.mean(val_scores_gamma, axis=1))
print(f"\n最佳 gamma 值: {param_range_gamma[best_gamma_idx]}, "
      f"验证分数: {np.mean(val_scores_gamma[best_gamma_idx]):.4f}")

# ============================================================
# 七、偏差-方差权衡可视化
# ============================================================

print("\n" + "=" * 60)
print("【偏差-方差权衡】")
print("偏差 (Bias):  模型对训练数据的拟合能力, 高偏差 = 欠拟合")
print("方差 (Variance): 模型对不同训练数据的敏感度, 高方差 = 过拟合")
print("")
print("验证曲线解读:")
print("  - 训练分数持续高, 验证分数低 => 高方差 (过拟合)")
print("  - 训练分数和验证分数都低 => 高偏差 (欠拟合)")
print("  - 两者都高且接近 => 偏差和方差都较低 (理想状态)")
print("  - 最优参数通常在验证曲线的峰值处")

# ============================================================
# 八、n_jobs 参数
# ============================================================

print("\n" + "=" * 60)
print("【n_jobs 参数】")
print("n_jobs 控制并行计算:")
print("  n_jobs=1   : 单进程 (默认)")
print("  n_jobs=-1  : 使用所有 CPU 核心")
print("  n_jobs=2   : 使用 2 个核心")
print("对于大数据集和复杂模型, 设置 n_jobs=-1 可显著加速")
print("注意: n_jobs=-1 在 Windows 上可能需要将代码放在 if __name__ == '__main__' 中")

# ============================================================
# 九、鸢尾花数据集学习曲线
# ============================================================

pipe_lr_iris = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, random_state=42))
])

ts_iris, tr_iris, va_iris = learning_curve(
    pipe_lr_iris, X_iris, y_iris,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv, n_jobs=-1
)

print("\n" + "=" * 60)
print("【鸢尾花数据集学习曲线 (LogisticRegression)】")
print(f"{'训练样本数':<12} {'训练分数':>12} {'验证分数':>12}")
print("-" * 38)
for i, size in enumerate(ts_iris):
    print(f"{size:<12} {np.mean(tr_iris[i]):>12.4f} {np.mean(va_iris[i]):>12.4f}")

# ============================================================
# 十、可视化
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. LogisticRegression 学习曲线
ax = axes[0, 0]
ax.fill_between(train_sizes_abs,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.15, color='steelblue')
ax.fill_between(train_sizes_abs,
                val_scores_mean - val_scores_std,
                val_scores_mean + val_scores_std,
                alpha=0.15, color='darkorange')
ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='steelblue',
        markersize=5, label='训练分数')
ax.plot(train_sizes_abs, val_scores_mean, 'o-', color='darkorange',
        markersize=5, label='验证分数')
ax.set_xlabel('训练样本数', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('学习曲线 (LogisticRegression)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 2. 过拟合/欠拟合诊断
for idx, (name, data) in enumerate(diagnose_data.items()):
    ax = axes[1, idx] if idx < 3 else axes[1, 2]
    ax.fill_between(data['train_sizes'],
                    data['train_mean'] - data['train_std'],
                    data['train_mean'] + data['train_std'],
                    alpha=0.15, color='steelblue')
    ax.fill_between(data['train_sizes'],
                    data['val_mean'] - data['val_std'],
                    data['val_mean'] + data['val_std'],
                    alpha=0.15, color='darkorange')
    ax.plot(data['train_sizes'], data['train_mean'], 'o-', color='steelblue',
            markersize=4, label='训练分数')
    ax.plot(data['train_sizes'], data['val_mean'], 'o-', color='darkorange',
            markersize=4, label='验证分数')
    ax.set_xlabel('训练样本数', fontsize=11)
    ax.set_ylabel('分数', fontsize=11)
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 3. 验证曲线 - C 参数
ax = axes[0, 1]
tr_C_mean = np.mean(train_scores_C, axis=1)
tr_C_std = np.std(train_scores_C, axis=1)
va_C_mean = np.mean(val_scores_C, axis=1)
va_C_std = np.std(val_scores_C, axis=1)
ax.semilogx(param_range_C, tr_C_mean, 'o-', color='steelblue', markersize=5, label='训练分数')
ax.semilogx(param_range_C, va_C_mean, 'o-', color='darkorange', markersize=5, label='验证分数')
ax.fill_between(param_range_C, tr_C_mean - tr_C_std, tr_C_mean + tr_C_std,
                alpha=0.15, color='steelblue')
ax.fill_between(param_range_C, va_C_mean - va_C_std, va_C_mean + va_C_std,
                alpha=0.15, color='darkorange')
ax.axvline(x=param_range_C[best_C_idx], color='red', linestyle='--', alpha=0.7,
           label=f'最佳C={param_range_C[best_C_idx]}')
ax.set_xlabel('C (对数尺度)', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('验证曲线 (SVC: C参数)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 4. 验证曲线 - gamma 参数
ax = axes[0, 2]
tr_g_mean = np.mean(train_scores_gamma, axis=1)
tr_g_std = np.std(train_scores_gamma, axis=1)
va_g_mean = np.mean(val_scores_gamma, axis=1)
va_g_std = np.std(val_scores_gamma, axis=1)
ax.semilogx(param_range_gamma, tr_g_mean, 'o-', color='steelblue', markersize=5, label='训练分数')
ax.semilogx(param_range_gamma, va_g_mean, 'o-', color='darkorange', markersize=5, label='验证分数')
ax.fill_between(param_range_gamma, tr_g_mean - tr_g_std, tr_g_mean + tr_g_std,
                alpha=0.15, color='steelblue')
ax.fill_between(param_range_gamma, va_g_mean - va_g_std, va_g_mean + va_g_std,
                alpha=0.15, color='darkorange')
ax.axvline(x=param_range_gamma[best_gamma_idx], color='red', linestyle='--', alpha=0.7,
           label=f'最佳gamma={param_range_gamma[best_gamma_idx]}')
ax.set_xlabel('gamma (对数尺度)', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('验证曲线 (SVC: gamma参数)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '学习曲线与验证曲线.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("学习曲线与验证曲线演示完成!")
