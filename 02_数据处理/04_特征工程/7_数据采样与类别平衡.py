# 数据来源: 自建不平衡数据集

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_imbalanced_data():
    """构建不平衡数据集"""
    np.random.seed(42)
    n_majority = 900
    n_minority = 100

    # 多数类
    X_majority = np.random.normal(loc=0, scale=1, size=(n_majority, 4))
    y_majority = np.zeros(n_majority, dtype=int)

    # 少数类
    X_minority = np.random.normal(loc=2, scale=1, size=(n_minority, 4))
    y_minority = np.ones(n_minority, dtype=int)

    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])

    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3", "feat_4"])
    df["target"] = y

    print("=== 不平衡数据集概览 ===")
    print(f"总样本数: {len(df)}")
    print(f"类别分布:\n{df['target'].value_counts()}")
    print(f"类别比例: {dict(df['target'].value_counts(normalize=True).round(3))}")
    print(f"\n数据前5行:\n{df.head()}")
    return df


def demo_random_sampling(df):
    """df.sample(): 随机采样"""
    print("\n" + "=" * 60)
    print("1. df.sample() 随机采样")
    print("=" * 60)

    # 随机采样 10%
    sample_10 = df.sample(frac=0.1, random_state=42)
    print(f"随机采样 10%: 原始{len(df)}行 → 采样{len(sample_10)}行")
    print(f"采样后类别分布:\n{sample_10['target'].value_counts()}")

    # 随机采样固定行数
    sample_n = df.sample(n=200, random_state=42)
    print(f"\n随机采样200行: 类别分布:\n{sample_n['target'].value_counts()}")

    # 有放回采样
    sample_replace = df.sample(n=1500, replace=True, random_state=42)
    print(f"\n有放回采样1500行: 类别分布:\n{sample_replace['target'].value_counts()}")

    # 按权重采样
    weights = np.where(df["target"] == 1, 9, 1)
    sample_weighted = df.sample(n=200, weights=weights, random_state=42)
    print(f"\n加权采样200行 (少数类权重×9): 类别分布:\n{sample_weighted['target'].value_counts()}")


def demo_stratified_sampling(df):
    """分层采样: 保持各类别比例"""
    print("\n" + "=" * 60)
    print("2. 分层采样")
    print("=" * 60)

    # 使用 train_test_split 的 stratify 参数
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"原始类别比例: {dict(y.value_counts(normalize=True).round(3))}")
    print(f"训练集类别比例: {dict(y_train.value_counts(normalize=True).round(3))}")
    print(f"测试集类别比例: {dict(y_test.value_counts(normalize=True).round(3))}")
    print("→ 分层采样保持各类别比例一致")

    # 不使用 stratify 的对比
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"\n非分层训练集类别比例: {dict(y_train2.value_counts(normalize=True).round(3))}")
    print("→ 非分层采样可能导致少数类比例偏移")

    # 分组采样: 每个类别采样相同数量
    min_count = df["target"].value_counts().min()
    stratified_sample = df.groupby("target").sample(n=min_count, random_state=42)
    print(f"\n等量采样 (每类{min_count}个): 类别分布:\n{stratified_sample['target'].value_counts()}")


def demo_oversampling_undersampling(df):
    """过采样与欠采样概念及手动实现"""
    print("\n" + "=" * 60)
    print("3. 过采样与欠采样")
    print("=" * 60)

    majority = df[df["target"] == 0]
    minority = df[df["target"] == 1]
    print(f"多数类: {len(majority)}, 少数类: {len(minority)}")

    # --- 欠采样: 减少多数类 ---
    majority_down = majority.sample(n=len(minority), random_state=42)
    df_undersampled = pd.concat([majority_down, minority]).sample(frac=1, random_state=42)
    print(f"\n欠采样后: {df_undersampled['target'].value_counts().to_dict()}")
    print(f"总样本数: {len(df_undersampled)} (原始: {len(df)})")

    # --- 过采样: 增加少数类 (复制) ---
    minority_up = minority.sample(n=len(majority), replace=True, random_state=42)
    df_oversampled = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
    print(f"\n过采样后 (复制): {df_oversampled['target'].value_counts().to_dict()}")
    print(f"总样本数: {len(df_oversampled)} (原始: {len(df)})")

    # --- 混合采样 ---
    target_majority = 450
    target_minority = 300
    majority_mixed = majority.sample(n=target_majority, random_state=42)
    minority_mixed = minority.sample(n=target_minority, replace=True, random_state=42)
    df_mixed = pd.concat([majority_mixed, minority_mixed]).sample(frac=1, random_state=42)
    print(f"\n混合采样后: {df_mixed['target'].value_counts().to_dict()}")
    print(f"总样本数: {len(df_mixed)}")

    print("\n采样策略对比:")
    print("  欠采样 - 减少多数类, 可能丢失信息, 训练快")
    print("  过采样 - 增加少数类, 保留多数类信息, 可能过拟合")
    print("  混合采样 - 折中方案, 兼顾信息量和类别平衡")


def demo_smote_concept(df):
    """SMOTE 概念说明及简单实现"""
    print("\n" + "=" * 60)
    print("4. SMOTE (合成少数类过采样) 概念")
    print("=" * 60)

    minority = df[df["target"] == 1].drop("target", axis=1).values
    print(f"少数类样本数: {len(minority)}")

    # 手动实现简单 SMOTE
    def simple_smote(X_minority, n_synthetic, k=5, random_state=42):
        """简单 SMOTE 实现: 在K近邻之间插值生成新样本"""
        rng = np.random.RandomState(random_state)
        n = len(X_minority)
        synthetic = []

        for _ in range(n_synthetic):
            idx = rng.randint(0, n)
            distances = np.sqrt(np.sum((X_minority - X_minority[idx]) ** 2, axis=1))
            distances[idx] = np.inf
            nn_indices = np.argsort(distances)[:k]
            nn_idx = rng.choice(nn_indices)
            alpha = rng.random()
            new_sample = X_minority[idx] + alpha * (X_minority[nn_idx] - X_minority[idx])
            synthetic.append(new_sample)

        return np.array(synthetic)

    n_to_generate = len(df[df["target"] == 0]) - len(minority)
    synthetic_samples = simple_smote(minority, n_to_generate, k=5, random_state=42)
    print(f"SMOTE 生成合成样本数: {len(synthetic_samples)}")
    print(f"合成样本前3行:\n{synthetic_samples[:3]}")

    # 合并数据
    feature_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
    df_synthetic = pd.DataFrame(synthetic_samples, columns=feature_cols)
    df_synthetic["target"] = 1
    df_smote = pd.concat([df, df_synthetic]).sample(frac=1, random_state=42)
    print(f"\nSMOTE 后类别分布: {df_smote['target'].value_counts().to_dict()}")

    # 使用 imbalanced-learn 库 (如果可用)
    print("\nimbalanced-learn 库使用示例:")
    print("  from imblearn.over_sampling import SMOTE")
    print("  smote = SMOTE(random_state=42)")
    print("  X_resampled, y_resampled = smote.fit_resample(X, y)")
    print("\nSMOTE 变体:")
    print("  SMOTE          - 基础版本, K近邻插值")
    print("  BorderlineSMOTE - 只在边界区域生成样本")
    print("  SVMSMOTE       - 用SVM确定边界区域")
    print("  ADASYN         - 根据难度自适应生成")
    print("  SMOTENC        - 支持分类特征")


def demo_class_weight(df):
    """class_weight 参数: 不采样也能处理不平衡"""
    print("\n" + "=" * 60)
    print("5. class_weight 参数处理不平衡")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 不使用 class_weight
    clf_default = LogisticRegression(max_iter=1000, random_state=42)
    clf_default.fit(X_train, y_train)
    y_pred_default = clf_default.predict(X_test)
    print("class_weight=None (默认):")
    print(classification_report(y_test, y_pred_default, target_names=["多数类(0)", "少数类(1)"]))

    # class_weight='balanced': 自动按类别频率的倒数加权
    clf_balanced = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_balanced.fit(X_train, y_train)
    y_pred_balanced = clf_balanced.predict(X_test)
    print("class_weight='balanced':")
    print(classification_report(y_test, y_pred_balanced, target_names=["多数类(0)", "少数类(1)"]))

    # 自定义权重
    clf_custom = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 9}, random_state=42)
    clf_custom.fit(X_train, y_train)
    y_pred_custom = clf_custom.predict(X_test)
    print("class_weight={0: 1, 1: 9}:")
    print(classification_report(y_test, y_pred_custom, target_names=["多数类(0)", "少数类(1)"]))

    # 混淆矩阵对比
    print("混淆矩阵对比:")
    print(f"  默认: \n{confusion_matrix(y_test, y_pred_default)}")
    print(f"  balanced: \n{confusion_matrix(y_test, y_pred_balanced)}")
    print(f"  自定义: \n{confusion_matrix(y_test, y_pred_custom)}")


def demo_imbalanced_visualization(df):
    """不平衡数据可视化"""
    print("\n" + "=" * 60)
    print("6. 不平衡数据可视化")
    print("=" * 60)

    # 类别分布统计
    counts = df["target"].value_counts()
    print(f"类别分布:\n{counts}")
    print(f"\n不平衡比率 (多数/少数): {counts[0] / counts[1]:.1f}:1")

    # 各类别特征统计
    print(f"\n各类别特征均值:")
    print(df.groupby("target").mean())

    print(f"\n各类别特征标准差:")
    print(df.groupby("target").std())

    # 可视化代码 (需要时取消注释)
    print("\n可视化代码示例:")
    print("  import matplotlib.pyplot as plt")
    print("  fig, axes = plt.subplots(1, 2, figsize=(12, 5))")
    print("  counts.plot(kind='bar', ax=axes[0], title='类别分布')")
    print("  df.groupby('target')['feat_1'].plot(kind='kde', ax=axes[1], legend=True)")
    print("  plt.savefig(OUTPUT_DIR / 'imbalanced_visualization.png')")


def main():
    df = build_imbalanced_data()

    demo_random_sampling(df)
    demo_stratified_sampling(df)
    demo_oversampling_undersampling(df)
    demo_smote_concept(df)
    demo_class_weight(df)
    demo_imbalanced_visualization(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  df.sample()          - 随机/加权/有放回采样")
    print("  分层采样              - 保持类别比例, train_test_split(stratify=y)")
    print("  欠采样               - 减少多数类, 简单但可能丢失信息")
    print("  过采样               - 复制少数类, 保留信息但可能过拟合")
    print("  SMOTE               - 合成少数类样本, 比简单复制更优")
    print("  class_weight         - 不改变数据, 通过权重调整损失函数")
    print("  实践建议: SMOTE + class_weight 结合使用效果更佳")
    print("=" * 60)


if __name__ == "__main__":
    main()
