# 数据来源: sklearn.datasets.load_breast_cancer (乳腺癌数据集), sklearn.datasets.load_iris (鸢尾花数据集)

"""
分类模型评估指标
================
本案例演示:
1. accuracy 准确率
2. precision 精确率
3. recall 召回率
4. f1-score F1分数
5. confusion_matrix 混淆矩阵
6. classification_report 分类报告
7. ROC 曲线与 AUC 分数
8. precision-recall 曲线
9. average_precision_score
10. 多分类指标 (macro/micro/weighted)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    PrecisionRecallDisplay
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、数据加载与准备 (乳腺癌 - 二分类)
# ============================================================

cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print("=" * 60)
print("【乳腺癌数据集信息】")
print(f"特征矩阵形状: {X_cancer.shape}")
print(f"类别名称: {cancer.target_names}")
print(f"  0 = {cancer.target_names[0]} (恶性)")
print(f"  1 = {cancer.target_names[1]} (良性)")
print(f"类别分布: 恶性={np.sum(y_cancer==0)}, 良性={np.sum(y_cancer==1)}")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.3, stratify=y_cancer, random_state=42
)

pipe_lr_c = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, random_state=42))
])
pipe_lr_c.fit(X_train_c, y_train_c)
y_pred_c = pipe_lr_c.predict(X_test_c)
y_prob_c = pipe_lr_c.predict_proba(X_test_c)[:, 1]

# ============================================================
# 二、accuracy 准确率
# ============================================================

print("\n" + "=" * 60)
print("【accuracy 准确率】")
acc = accuracy_score(y_test_c, y_pred_c)
print(f"准确率: {acc:.4f}")
print("准确率 = 正确预测数 / 总样本数")
print("注意: 类别不平衡时, 准确率可能具有误导性")

# ============================================================
# 三、precision 精确率
# ============================================================

print("\n" + "=" * 60)
print("【precision 精确率】")
prec = precision_score(y_test_c, y_pred_c)
print(f"精确率: {prec:.4f}")
print("精确率 = TP / (TP + FP)")
print("含义: 预测为正的样本中, 真正为正的比例")
print("在医疗诊断中: 预测为恶性肿瘤的样本中, 真正是恶性肿瘤的比例")

# ============================================================
# 四、recall 召回率
# ============================================================

print("\n" + "=" * 60)
print("【recall 召回率】")
rec = recall_score(y_test_c, y_pred_c)
print(f"召回率: {rec:.4f}")
print("召回率 = TP / (TP + FN)")
print("含义: 真正为正的样本中, 被正确预测为正的比例")
print("在医疗诊断中: 真正是恶性肿瘤的样本中, 被正确识别的比例")
print("医疗场景中召回率通常更重要: 漏诊(假阴性)的代价远大于误诊(假阳性)")

# ============================================================
# 五、f1-score F1分数
# ============================================================

print("\n" + "=" * 60)
print("【f1-score F1分数】")
f1 = f1_score(y_test_c, y_pred_c)
print(f"F1分数: {f1:.4f}")
print("F1 = 2 * (Precision * Recall) / (Precision + Recall)")
print("F1 是精确率和召回率的调和平均, 综合衡量模型性能")
print(f"验证: 2 * ({prec:.4f} * {rec:.4f}) / ({prec:.4f} + {rec:.4f}) = "
      f"{2 * prec * rec / (prec + rec):.4f}")

# ============================================================
# 六、confusion_matrix 混淆矩阵
# ============================================================

print("\n" + "=" * 60)
print("【confusion_matrix 混淆矩阵】")
cm = confusion_matrix(y_test_c, y_pred_c)
print(f"混淆矩阵:\n{cm}")
print(f"  TN (真阴性, 恶性正确识别): {cm[0, 0]}")
print(f"  FP (假阳性, 良性误判恶性): {cm[0, 1]}")
print(f"  FN (假阴性, 恶性误判良性): {cm[1, 0]}")
print(f"  TP (真阳性, 良性正确识别): {cm[1, 1]}")

tn, fp, fn, tp = cm.ravel()
print(f"\n从混淆矩阵推导:")
print(f"  准确率 = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp}+{tn}+{fp}+{fn}) = "
      f"{(tp+tn)/(tp+tn+fp+fn):.4f}")
print(f"  精确率 = TP/(TP+FP) = {tp}/({tp}+{fp}) = {tp/(tp+fp):.4f}")
print(f"  召回率 = TP/(TP+FN) = {tp}/({tp}+{fn}) = {tp/(tp+fn):.4f}")

# ============================================================
# 七、classification_report 分类报告
# ============================================================

print("\n" + "=" * 60)
print("【classification_report 分类报告】")
print(classification_report(y_test_c, y_pred_c, target_names=cancer.target_names))
print("support: 每个类别的实际样本数")
print("macro avg: 各类别指标的简单平均")
print("weighted avg: 按各类别样本数加权的平均")

# ============================================================
# 八、ROC 曲线与 AUC 分数
# ============================================================

print("\n" + "=" * 60)
print("【ROC 曲线与 AUC 分数】")

fpr, tpr, thresholds = roc_curve(y_test_c, y_prob_c)
auc_score = roc_auc_score(y_test_c, y_prob_c)

print(f"AUC 分数: {auc_score:.4f}")
print("ROC 曲线: 以 FPR (假正率) 为横轴, TPR (真正率) 为纵轴")
print("AUC: ROC 曲线下面积, 越接近1越好")
print("  AUC = 1.0: 完美分类器")
print("  AUC = 0.5: 随机分类器 (无区分能力)")
print("  AUC < 0.5: 比随机还差 (可能标签反转)")

best_threshold_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[best_threshold_idx]
print(f"\n最优阈值 (Youden's J): {best_threshold:.4f}")
print(f"  对应 TPR: {tpr[best_threshold_idx]:.4f}")
print(f"  对应 FPR: {fpr[best_threshold_idx]:.4f}")

# ============================================================
# 九、precision-recall 曲线
# ============================================================

print("\n" + "=" * 60)
print("【precision-recall 曲线】")

prec_values, rec_values, pr_thresholds = precision_recall_curve(y_test_c, y_prob_c)
ap_score = average_precision_score(y_test_c, y_prob_c)

print(f"Average Precision (AP): {ap_score:.4f}")
print("PR 曲线: 以 Recall 为横轴, Precision 为纵轴")
print("AP: PR 曲线下面积, 对类别不平衡问题比 AUC 更具信息量")
print(f"PR 曲线阈值数量: {len(pr_thresholds)}")

f1_pr = 2 * prec_values * rec_values / (prec_values + rec_values + 1e-8)
best_pr_idx = np.argmax(f1_pr)
best_pr_threshold = pr_thresholds[best_pr_idx] if best_pr_idx < len(pr_thresholds) else pr_thresholds[-1]
print(f"最优阈值 (F1最大): {best_pr_threshold:.4f}")
print(f"  对应 Precision: {prec_values[best_pr_idx]:.4f}")
print(f"  对应 Recall: {rec_values[best_pr_idx]:.4f}")
print(f"  对应 F1: {f1_pr[best_pr_idx]:.4f}")

# ============================================================
# 十、多分类指标 (macro/micro/weighted) - 鸢尾花数据集
# ============================================================

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.3, stratify=y_iris, random_state=42
)

pipe_lr_i = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, random_state=42))
])
pipe_lr_i.fit(X_train_i, y_train_i)
y_pred_i = pipe_lr_i.predict(X_test_i)
y_prob_i = pipe_lr_i.predict_proba(X_test_i)

print("\n" + "=" * 60)
print("【多分类指标 - 鸢尾花数据集】")
print(f"类别名称: {iris.target_names}")

print("\n--- 各平均方式对比 ---")
for avg in ['macro', 'micro', 'weighted']:
    p = precision_score(y_test_i, y_pred_i, average=avg)
    r = recall_score(y_test_i, y_pred_i, average=avg)
    f = f1_score(y_test_i, y_pred_i, average=avg)
    print(f"  average='{avg:<10} => precision={p:.4f}, recall={r:.4f}, f1={f:.4f}")

print("\n说明:")
print("  macro:    各类别指标的简单平均, 不考虑类别大小")
print("  micro:    全局计算 TP/FP/FN, 适合关注整体表现")
print("  weighted: 按各类别样本数加权平均, 考虑类别不平衡")
print("  注意: 多分类中 micro-F1 == micro-precision == micro-recall == accuracy")

micro_acc = accuracy_score(y_test_i, y_pred_i)
micro_f1 = f1_score(y_test_i, y_pred_i, average='micro')
print(f"\n验证: accuracy={micro_acc:.4f}, micro-F1={micro_f1:.4f} (应相等)")

print("\n--- 各类别详细指标 ---")
print(classification_report(y_test_i, y_pred_i, target_names=iris.target_names))

# 多分类混淆矩阵
cm_multi = confusion_matrix(y_test_i, y_pred_i)
print("多分类混淆矩阵:")
print(cm_multi)

# 多分类 ROC-AUC
y_test_bin = label_binarize(y_test_i, classes=[0, 1, 2])
auc_macro = roc_auc_score(y_test_bin, y_prob_i, multi_class='ovr', average='macro')
auc_weighted = roc_auc_score(y_test_bin, y_prob_i, multi_class='ovr', average='weighted')
print(f"\n多分类 AUC (OvR):")
print(f"  macro AUC:    {auc_macro:.4f}")
print(f"  weighted AUC: {auc_weighted:.4f}")

# ============================================================
# 十一、不同阈值对指标的影响
# ============================================================

print("\n" + "=" * 60)
print("【不同阈值对分类指标的影响 (乳腺癌)】")
print(f"{'阈值':<8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'预测正例数':>10}")
print("-" * 44)

thresholds_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for th in thresholds_test:
    y_pred_th = (y_prob_c >= th).astype(int)
    p_th = precision_score(y_test_c, y_pred_th, zero_division=0)
    r_th = recall_score(y_test_c, y_pred_th, zero_division=0)
    f1_th = f1_score(y_test_c, y_pred_th, zero_division=0)
    n_pos = np.sum(y_pred_th)
    print(f"{th:<8.1f} {p_th:>8.4f} {r_th:>8.4f} {f1_th:>8.4f} {n_pos:>10}")

print("\n阈值越低 => 召回率越高, 精确率越低 (更多样本被判为正)")
print("阈值越高 => 精确率越高, 召回率越低 (更少样本被判为正)")

# ============================================================
# 十二、可视化
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. 混淆矩阵 (二分类)
ax = axes[0, 0]
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=cancer.target_names, yticklabels=cancer.target_names,
       title='混淆矩阵 (乳腺癌)', xlabel='预测类别', ylabel='真实类别')
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color=color, fontsize=16)

# 2. ROC 曲线
ax = axes[0, 1]
ax.plot(fpr, tpr, color='steelblue', linewidth=2,
        label=f'ROC (AUC={auc_score:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类器')
ax.plot(fpr[best_threshold_idx], tpr[best_threshold_idx], 'ro', markersize=8,
        label=f'最优阈值={best_threshold:.3f}')
ax.set_xlabel('假正率 (FPR)', fontsize=12)
ax.set_ylabel('真正率 (TPR)', fontsize=12)
ax.set_title('ROC 曲线 (乳腺癌)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Precision-Recall 曲线
ax = axes[0, 2]
ax.plot(rec_values, prec_values, color='darkorange', linewidth=2,
        label=f'PR (AP={ap_score:.3f})')
ax.plot(rec_values[best_pr_idx], prec_values[best_pr_idx], 'ro', markersize=8,
        label=f'最优阈值={best_pr_threshold:.3f}')
baseline = np.sum(y_test_c == 1) / len(y_test_c)
ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5,
           label=f'基线 (正例比例={baseline:.3f})')
ax.set_xlabel('召回率 (Recall)', fontsize=12)
ax.set_ylabel('精确率 (Precision)', fontsize=12)
ax.set_title('Precision-Recall 曲线 (乳腺癌)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 4. 多分类混淆矩阵 (鸢尾花)
ax = axes[1, 0]
im2 = ax.imshow(cm_multi, interpolation='nearest', cmap=plt.cm.Blues)
fig.colorbar(im2, ax=ax)
ax.set(xticks=np.arange(3), yticks=np.arange(3),
       xticklabels=iris.target_names, yticklabels=iris.target_names,
       title='混淆矩阵 (鸢尾花)', xlabel='预测类别', ylabel='真实类别')
for i in range(3):
    for j in range(3):
        color = "white" if cm_multi[i, j] > cm_multi.max() / 2 else "black"
        ax.text(j, i, format(cm_multi[i, j], 'd'), ha="center", va="center",
                color=color, fontsize=14)

# 5. 多分类 ROC 曲线
ax = axes[1, 1]
colors = ['steelblue', 'darkorange', 'green']
for i, cls_name in enumerate(iris.target_names):
    fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob_i[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], y_prob_i[:, i])
    ax.plot(fpr_i, tpr_i, color=colors[i], linewidth=2,
            label=f'{cls_name} (AUC={auc_i:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类器')
ax.set_xlabel('假正率 (FPR)', fontsize=12)
ax.set_ylabel('真正率 (TPR)', fontsize=12)
ax.set_title('多分类 ROC 曲线 (鸢尾花, OvR)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 6. 不同阈值下的指标变化
ax = axes[1, 2]
prec_list = []
rec_list = []
f1_list = []
for th in np.linspace(0.05, 0.95, 50):
    y_pred_th = (y_prob_c >= th).astype(int)
    prec_list.append(precision_score(y_test_c, y_pred_th, zero_division=0))
    rec_list.append(recall_score(y_test_c, y_pred_th, zero_division=0))
    f1_list.append(f1_score(y_test_c, y_pred_th, zero_division=0))

th_range = np.linspace(0.05, 0.95, 50)
ax.plot(th_range, prec_list, color='steelblue', linewidth=2, label='精确率')
ax.plot(th_range, rec_list, color='darkorange', linewidth=2, label='召回率')
ax.plot(th_range, f1_list, color='green', linewidth=2, label='F1分数')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='默认阈值=0.5')
ax.set_xlabel('分类阈值', fontsize=12)
ax.set_ylabel('指标值', fontsize=12)
ax.set_title('阈值对指标的影响 (乳腺癌)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '分类模型评估指标.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("分类模型评估指标演示完成!")
