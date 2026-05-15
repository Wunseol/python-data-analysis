# 依赖库最低版本要求: scikit-learn>=1.3, numpy>=1.24, matplotlib>=3.7, pandas>=2.0
# 数据来源: sklearn.datasets.load_wine (红酒数据集), sklearn.datasets.load_iris (鸢尾花数据集)

"""
超参数调优
==========
本案例演示:
1. GridSearchCV 网格搜索
2. RandomizedSearchCV 随机搜索
3. 参数分布与 n_iter 参数
4. scoring 评分参数
5. best_params_ / best_score_ / cv_results_
6. 网格搜索 vs 随机搜索对比
7. 超参数调优实践技巧
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、数据加载与准备
# ============================================================

wine = load_wine()
X = wine.data
y = wine.target

print("=" * 60)
print("【红酒数据集信息】")
print(f"特征矩阵形状: {X.shape}")
print(f"类别名称: {wine.target_names}")
print(f"类别分布: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 二、GridSearchCV 网格搜索 (RandomForest)
# ============================================================

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [3, 5, 10, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

total_combos = 1
for v in param_grid_rf.values():
    total_combos *= len(v)
print(f"\n网格搜索参数组合总数: {total_combos}")

grid_rf = GridSearchCV(
    pipe_rf,
    param_grid_rf,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True,
    verbose=0
)
grid_rf.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【GridSearchCV 网格搜索结果 (RandomForest)】")
print(f"最佳参数: {grid_rf.best_params_}")
print(f"最佳交叉验证分数: {grid_rf.best_score_:.4f}")
print(f"测试集准确率: {grid_rf.score(X_test, y_test):.4f}")

# ============================================================
# 三、best_params_ / best_score_ / cv_results_ 详解
# ============================================================

print("\n" + "=" * 60)
print("【best_params_ / best_score_ / cv_results_ 详解】")
print(f"best_params_  : {grid_rf.best_params_}")
print(f"best_score_   : {grid_rf.best_score_:.4f}")
print(f"best_estimator_: {grid_rf.best_estimator_}")

cv_results = grid_rf.cv_results_
df_results = pd.DataFrame({
    'params': cv_results['params'],
    'mean_test_score': cv_results['mean_test_score'],
    'std_test_score': cv_results['std_test_score'],
    'mean_train_score': cv_results['mean_train_score'],
    'rank_test_score': cv_results['rank_test_score']
})

print(f"\ncv_results_ 前5名:")
top5 = df_results.nsmallest(5, 'rank_test_score')
for idx, row in top5.iterrows():
    print(f"  排名{row['rank_test_score']:.0f}: "
          f"验证={row['mean_test_score']:.4f}±{row['std_test_score']:.4f}, "
          f"训练={row['mean_train_score']:.4f}, "
          f"参数={row['params']}")

# ============================================================
# 四、GridSearchCV 网格搜索 (SVC)
# ============================================================

pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(random_state=42))
])

param_grid_svc = {
    'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'clf__kernel': ['linear', 'rbf', 'poly'],
    'clf__gamma': ['scale', 'auto']
}

grid_svc = GridSearchCV(
    pipe_svc,
    param_grid_svc,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)
grid_svc.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【GridSearchCV 网格搜索结果 (SVC)】")
print(f"最佳参数: {grid_svc.best_params_}")
print(f"最佳交叉验证分数: {grid_svc.best_score_:.4f}")
print(f"测试集准确率: {grid_svc.score(X_test, y_test):.4f}")

# ============================================================
# 五、RandomizedSearchCV 随机搜索
# ============================================================

pipe_rf_rand = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_dist_rf = {
    'clf__n_estimators': randint(50, 300),
    'clf__max_depth': [3, 5, 10, 15, 20, None],
    'clf__min_samples_split': randint(2, 20),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__max_features': ['sqrt', 'log2', None]
}

random_rf = RandomizedSearchCV(
    pipe_rf_rand,
    param_dist_rf,
    n_iter=50,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)
random_rf.fit(X_train, y_train)

print("\n" + "=" * 60)
print("【RandomizedSearchCV 随机搜索结果 (RandomForest)】")
print(f"n_iter=50, 搜索50组随机参数组合")
print(f"最佳参数: {random_rf.best_params_}")
print(f"最佳交叉验证分数: {random_rf.best_score_:.4f}")
print(f"测试集准确率: {random_rf.score(X_test, y_test):.4f}")

# ============================================================
# 六、参数分布与 n_iter 参数
# ============================================================

print("\n" + "=" * 60)
print("【参数分布与 n_iter 参数】")
print("参数分布类型:")
print("  randint(low, high)       : 均匀离散整数分布")
print("  uniform(low, high)       : 均匀连续分布")
print("  loguniform(a, b)         : 对数均匀分布 (适合学习率、C等跨越多个数量级的参数)")
print("")
print("n_iter 参数:")
print("  控制随机搜索的迭代次数 (采样参数组合数)")
print("  n_iter 越大, 搜索越充分, 但耗时越长")
print("  经验: n_iter 通常设为参数组合总数的 10%~50%")

n_iter_values = [10, 20, 50, 100]
n_iter_scores = []
n_iter_times = []

import time

for n_iter in n_iter_values:
    start = time.time()
    rs = RandomizedSearchCV(
        pipe_rf_rand, param_dist_rf,
        n_iter=n_iter, cv=cv, scoring='accuracy',
        n_jobs=-1, random_state=42
    )
    rs.fit(X_train, y_train)
    elapsed = time.time() - start
    n_iter_scores.append(rs.best_score_)
    n_iter_times.append(elapsed)
    print(f"  n_iter={n_iter:<4} => 最佳分数={rs.best_score_:.4f}, 耗时={elapsed:.2f}s")

# ============================================================
# 七、scoring 评分参数
# ============================================================

print("\n" + "=" * 60)
print("【scoring 评分参数】")
print("scoring 支持多种评分方式:")
print("  'accuracy'       : 准确率 (默认, 分类)")
print("  'f1'             : F1分数 (二分类)")
print("  'f1_macro'       : 宏平均F1 (多分类)")
print("  'f1_weighted'    : 加权F1 (多分类)")
print("  'precision'      : 精确率")
print("  'recall'         : 召回率")
print("  'roc_auc'        : ROC-AUC (需要概率输出)")
print("  'neg_log_loss'   : 负对数损失 (需要概率输出)")
print("")

scoring_methods = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
for scoring in scoring_methods:
    gs_temp = GridSearchCV(
        pipe_svc,
        {'clf__C': [0.1, 1.0, 10.0], 'clf__kernel': ['linear', 'rbf']},
        cv=cv, scoring=scoring, n_jobs=-1
    )
    gs_temp.fit(X_train, y_train)
    print(f"  scoring='{scoring:<18} => 最佳分数={gs_temp.best_score_:.4f}, "
          f"最佳参数={gs_temp.best_params_}")

# ============================================================
# 八、网格搜索 vs 随机搜索对比
# ============================================================

print("\n" + "=" * 60)
print("【网格搜索 vs 随机搜索对比】")

pipe_compare = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(random_state=42))
])

param_grid_compare = {
    'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'clf__gamma': [0.001, 0.01, 0.1, 1.0, 10.0]
}

grid_total = 1
for v in param_grid_compare.values():
    grid_total *= len(v)
print(f"网格搜索参数组合数: {grid_total}")

start_grid = time.time()
grid_compare = GridSearchCV(
    pipe_compare, param_grid_compare, cv=cv, scoring='accuracy', n_jobs=-1
)
grid_compare.fit(X_train, y_train)
time_grid = time.time() - start_grid

param_dist_compare = {
    'clf__C': loguniform(1e-3, 1e2),
    'clf__gamma': loguniform(1e-3, 1e1)
}

start_random = time.time()
random_compare = RandomizedSearchCV(
    pipe_compare, param_dist_compare, n_iter=20, cv=cv,
    scoring='accuracy', n_jobs=-1, random_state=42
)
random_compare.fit(X_train, y_train)
time_random = time.time() - start_random

print(f"\n网格搜索:   组合数={grid_total}, 最佳分数={grid_compare.best_score_:.4f}, "
      f"最佳参数={grid_compare.best_params_}, 耗时={time_grid:.2f}s")
print(f"随机搜索:   n_iter=20,  最佳分数={random_compare.best_score_:.4f}, "
      f"最佳参数={random_compare.best_params_}, 耗时={time_random:.2f}s")
print(f"随机搜索仅用 {20/grid_total*100:.1f}% 的计算量就达到了可比的效果")

# ============================================================
# 九、超参数调优实践技巧
# ============================================================

print("\n" + "=" * 60)
print("【超参数调优实践技巧】")
print("1. 粗调再细调: 先用较大步长粗略搜索, 再在最优区域精细搜索")
print("2. 优先调重要参数: 如 n_estimators, C, max_depth 等影响较大的参数")
print("3. 使用对数尺度: C, gamma, learning_rate 等参数适合用 loguniform")
print("4. 随机搜索更高效: 参数空间大时, RandomizedSearchCV 通常更高效")
print("5. 注意数据泄露: 始终在 Pipeline 中调参, 避免预处理时泄露测试集信息")
print("6. 交叉验证折数: 通常 cv=5 即可, 数据少时可增大到 10")
print("7. 并行加速: 设置 n_jobs=-1 使用所有 CPU 核心")
print("8. scoring 选择: 根据业务需求选择合适的评分指标")

# 粗调再细调示例
print("\n--- 粗调再细调示例 ---")
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', random_state=42))
])

param_grid_coarse = {
    'clf__C': [0.1, 1.0, 10.0, 100.0],
    'clf__gamma': [0.01, 0.1, 1.0, 10.0]
}
grid_coarse = GridSearchCV(pipe_svm, param_grid_coarse, cv=cv, scoring='accuracy', n_jobs=-1)
grid_coarse.fit(X_train, y_train)
print(f"粗调最佳: C={grid_coarse.best_params_['clf__C']}, "
      f"gamma={grid_coarse.best_params_['clf__gamma']}, "
      f"分数={grid_coarse.best_score_:.4f}")

best_C = grid_coarse.best_params_['clf__C']
best_gamma = grid_coarse.best_params_['clf__gamma']

param_grid_fine = {
    'clf__C': [best_C * 0.5, best_C, best_C * 2],
    'clf__gamma': [best_gamma * 0.5, best_gamma, best_gamma * 2]
}
grid_fine = GridSearchCV(pipe_svm, param_grid_fine, cv=cv, scoring='accuracy', n_jobs=-1)
grid_fine.fit(X_train, y_train)
print(f"细调最佳: C={grid_fine.best_params_['clf__C']}, "
      f"gamma={grid_fine.best_params_['clf__gamma']}, "
      f"分数={grid_fine.best_score_:.4f}")

# ============================================================
# 十、最终模型评估
# ============================================================

best_model = grid_fine.best_estimator_
y_pred = best_model.predict(X_test)

print("\n" + "=" * 60)
print("【最终模型评估 (细调后SVC)】")
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n【分类报告】")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# ============================================================
# 十一、可视化
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. cv_results_ 热力图 (SVC GridSearch)
df_svc = pd.DataFrame(grid_svc.cv_results_)
pivot_data = df_svc.pivot_table(
    index='param_clf__C',
    columns='param_clf__kernel',
    values='mean_test_score'
)
im = axes[0, 0].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
axes[0, 0].set_xticks(range(len(pivot_data.columns)))
axes[0, 0].set_xticklabels(pivot_data.columns)
axes[0, 0].set_yticks(range(len(pivot_data.index)))
axes[0, 0].set_yticklabels(pivot_data.index)
axes[0, 0].set_xlabel('kernel')
axes[0, 0].set_ylabel('C')
axes[0, 0].set_title('GridSearchCV 热力图 (SVC)', fontsize=13)
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        val = pivot_data.values[i, j]
        axes[0, 0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10)
fig.colorbar(im, ax=axes[0, 0])

# 2. n_iter 对随机搜索的影响
ax2 = axes[0, 1]
line1 = ax2.plot(n_iter_values, n_iter_scores, 'o-', color='steelblue', markersize=8, label='最佳分数')
ax2.set_xlabel('n_iter', fontsize=12)
ax2.set_ylabel('最佳交叉验证分数', fontsize=12, color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(n_iter_values, n_iter_times, 's--', color='darkorange', markersize=8, label='耗时')
ax2_twin.set_ylabel('耗时 (秒)', fontsize=12, color='darkorange')
ax2_twin.tick_params(axis='y', labelcolor='darkorange')
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=10)
ax2.set_title('n_iter 对随机搜索的影响', fontsize=13)
ax2.grid(True, alpha=0.3)

# 3. 网格搜索 vs 随机搜索
methods = ['网格搜索\n(全量)', '随机搜索\n(n_iter=20)']
scores_compare = [grid_compare.best_score_, random_compare.best_score_]
times_compare = [time_grid, time_random]

bars = axes[1, 0].bar(methods, scores_compare, color=['steelblue', 'darkorange'], alpha=0.8, width=0.5)
axes[1, 0].set_ylabel('最佳交叉验证分数', fontsize=12)
axes[1, 0].set_title('网格搜索 vs 随机搜索 (分数)', fontsize=13)
for bar, score in zip(bars, scores_compare):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{score:.4f}', ha='center', fontsize=11)
axes[1, 0].set_ylim(min(scores_compare) - 0.02, max(scores_compare) + 0.02)

# 4. 粗调 vs 细调
stages = ['粗调', '细调']
stage_scores = [grid_coarse.best_score_, grid_fine.best_score_]
bars2 = axes[1, 1].bar(stages, stage_scores, color=['steelblue', 'green'], alpha=0.8, width=0.4)
axes[1, 1].set_ylabel('最佳交叉验证分数', fontsize=12)
axes[1, 1].set_title('粗调 vs 细调', fontsize=13)
for bar, score in zip(bars2, stage_scores):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{score:.4f}', ha='center', fontsize=11)
axes[1, 1].set_ylim(min(stage_scores) - 0.02, max(stage_scores) + 0.02)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '超参数调优.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("超参数调优演示完成!")
