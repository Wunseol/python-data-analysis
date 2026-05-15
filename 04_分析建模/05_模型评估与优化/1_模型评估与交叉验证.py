# 数据来源: sklearn.datasets.load_wine (红酒数据集)

"""
模型评估与交叉验证
==================
本案例演示:
1. cross_val_score 交叉验证评分
2. KFold / StratifiedKFold 分层交叉验证
3. GridSearchCV 网格搜索调参
4. classification_report 分类报告
5. confusion_matrix 混淆矩阵
6. ROC-AUC 概念
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 一、数据加载与划分
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

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 二、cross_val_score 交叉验证
# ============================================================

models = {
    '逻辑回归': LogisticRegression(max_iter=200, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\n" + "=" * 60)
print("【5折交叉验证对比】")
print(f"{'模型':<15} {'均值':>8} {'标准差':>8} {'各折分数'}")
print("-" * 70)

for name, model in models.items():
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"{name:<15} {scores.mean():>8.4f} {scores.std():>8.4f} {scores}")

# ============================================================
# 三、KFold vs StratifiedKFold
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr = LogisticRegression(max_iter=200, random_state=42)

kf_scores = cross_val_score(lr, X_train_s, y_train, cv=kf, scoring='accuracy')
skf_scores = cross_val_score(lr, X_train_s, y_train, cv=skf, scoring='accuracy')

print("\n" + "=" * 60)
print("【KFold vs StratifiedKFold (逻辑回归)】")
print(f"KFold          : 均值={kf_scores.mean():.4f}, 标准差={kf_scores.std():.4f}")
print(f"StratifiedKFold: 均值={skf_scores.mean():.4f}, 标准差={skf_scores.std():.4f}")
print("StratifiedKFold 保证每折中类别比例一致, 通常更稳定")

# ============================================================
# 四、GridSearchCV 网格搜索调参
# ============================================================

# 对 SVM 进行网格搜索
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True
)
grid_search.fit(X_train_s, y_train)

print("\n" + "=" * 60)
print("【GridSearchCV 网格搜索结果】")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
print(f"最佳模型: {grid_search.best_estimator_}")

print(f"\n所有参数组合结果:")
print(f"{'参数':<40} {'验证均值':>10} {'验证标准差':>10} {'排名':>6}")
print("-" * 70)
results = grid_search.cv_results_
for i in range(len(results['params'])):
    params_str = str(results['params'][i])
    mean_score = results['mean_test_score'][i]
    std_score = results['std_test_score'][i]
    rank = results['rank_test_score'][i]
    print(f"{params_str:<40} {mean_score:>10.4f} {std_score:>10.4f} {rank:>6}")

# ============================================================
# 五、最优模型评估
# ============================================================

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_s)

print("\n" + "=" * 60)
print("【最优模型测试集评估】")
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n【分类报告】")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("【混淆矩阵】")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================
# 六、ROC-AUC 概念
# ============================================================

# 多分类 ROC-AUC: 使用 One-vs-Rest 策略
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# 使用概率输出的 SVM
svm_proba = SVC(kernel=grid_search.best_params_['kernel'],
                 C=grid_search.best_params_['C'],
                 gamma=grid_search.best_params_['gamma'],
                 probability=True, random_state=42)
svm_proba.fit(X_train_s, y_train)
y_score = svm_proba.predict_proba(X_test_s)

# 计算宏平均和微平均 AUC
auc_macro = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='macro')
auc_micro = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='micro')

print("\n" + "=" * 60)
print("【ROC-AUC 概念】")
print("ROC 曲线: 真正率 (TPR) vs 假正率 (FPR) 的曲线")
print("AUC: ROC 曲线下面积, 越接近1越好")
print(f"宏平均 AUC: {auc_macro:.4f}")
print(f"微平均 AUC: {auc_micro:.4f}")

# 各类别 AUC
for i, cls_name in enumerate(wine.target_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    auc_cls = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    print(f"  类别 {cls_name}: AUC = {auc_cls:.4f}")

# ============================================================
# 七、可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 交叉验证对比
model_names = list(models.keys())
cv_means = []
cv_stds = []
for name, model in models.items():
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

bars = axes[0].bar(model_names, cv_means, yerr=cv_stds, capsize=5,
                    color=['steelblue', 'darkorange', 'green'], alpha=0.8)
axes[0].set_ylabel('准确率', fontsize=12)
axes[0].set_title('5折交叉验证对比', fontsize=13)
axes[0].set_ylim(0.9, 1.0)
for bar, mean in zip(bars, cv_means):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{mean:.4f}', ha='center', fontsize=11)

# 2. 混淆矩阵
im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].figure.colorbar(im, ax=axes[1])
axes[1].set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=wine.target_names,
            yticklabels=wine.target_names,
            title='混淆矩阵 (最优SVM)',
            xlabel='预测类别',
            ylabel='真实类别')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1].text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)

# 3. ROC 曲线
colors = ['steelblue', 'darkorange', 'green']
for i, cls_name in enumerate(wine.target_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    auc_cls = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    axes[2].plot(fpr, tpr, color=colors[i], linewidth=2,
                 label=f'{cls_name} (AUC={auc_cls:.3f})')

axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类器')
axes[2].set_xlabel('假正率 (FPR)', fontsize=12)
axes[2].set_ylabel('真正率 (TPR)', fontsize=12)
axes[2].set_title('ROC 曲线 (One-vs-Rest)', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('模型评估与交叉验证.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("模型评估与交叉验证演示完成!")
