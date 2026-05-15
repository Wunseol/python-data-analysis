# 数据来源: sklearn.datasets.load_breast_cancer (乳腺癌数据集)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    chi2,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载乳腺癌数据集"""
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df["target"] = cancer.target
    print("=== 乳腺癌数据集概览 ===")
    print(f"特征数: {cancer.data.shape[1]}, 样本数: {cancer.data.shape[0]}")
    print(f"类别分布: {dict(zip(*np.unique(cancer.target, return_counts=True)))}")
    print(f"\n特征名 (前10): {cancer.feature_names[:10].tolist()}")
    print(f"\n数据前3行:\n{df.head(3)}")
    return df


def demo_variance_threshold(df):
    """VarianceThreshold: 移除低方差特征"""
    print("\n" + "=" * 60)
    print("1. VarianceThreshold 方差阈值法")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]

    # 查看各特征方差
    variances = X.var()
    print(f"方差统计:\n{variances.describe()}")
    print(f"\n方差最小的5个特征:")
    print(variances.nsmallest(5))

    # 默认阈值=0: 移除零方差特征
    vt0 = VarianceThreshold(threshold=0)
    X_vt0 = vt0.fit_transform(X)
    print(f"\nthreshold=0: {X.shape[1]} → {X_vt0.shape[1]} 个特征 (移除零方差)")

    # 自定义阈值
    threshold_median = variances.median()
    vt_med = VarianceThreshold(threshold=threshold_median)
    X_vt_med = vt_med.fit_transform(X)
    print(f"threshold={threshold_median:.2f} (中位数): {X.shape[1]} → {X_vt_med.shape[1]} 个特征")

    # 查看保留/移除的特征
    retained = vt_med.get_feature_names_out()
    removed = [c for c in X.columns if c not in retained]
    print(f"\n保留的特征 ({len(retained)}个): {retained[:5].tolist()}...")
    print(f"移除的特征 ({len(removed)}个): {removed[:5].tolist()}...")

    # 注意: 方差受量纲影响, 应先标准化再使用
    print("\n注意: 方差阈值法受特征量纲影响, 建议先标准化再使用")


def demo_correlation_selection(df):
    """基于相关性的特征选择"""
    print("\n" + "=" * 60)
    print("2. 基于相关性的特征选择")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]

    # 特征与目标的相关系数
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"与目标相关系数 (前10):\n{corr_with_target.head(10)}")

    # 选择相关系数 > 0.5 的特征
    high_corr_features = corr_with_target[corr_with_target > 0.5].index.tolist()
    print(f"\n相关系数 > 0.5 的特征 ({len(high_corr_features)}个): {high_corr_features[:5]}...")

    # 特征间相关性: 移除高度相关的冗余特征
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, row, upper.loc[col, row])
                       for col in upper.columns for row in upper.index
                       if upper.loc[col, row] > 0.95]
    print(f"\n特征间相关系数 > 0.95 的对数: {len(high_corr_pairs)}")
    for c1, c2, v in high_corr_pairs[:5]:
        print(f"  {c1} ↔ {c2}: {v:.4f}")

    # 移除冗余特征 (保留与目标相关性更高的)
    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > 0.95].index.tolist()
        for hc in high_corr:
            if corr_with_target.get(hc, 0) < corr_with_target.get(col, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)
    print(f"\n建议移除的冗余特征 ({len(to_drop)}个): {list(to_drop)[:5]}...")


def demo_select_kbest(df):
    """SelectKBest: 基于统计检验选择Top-K特征"""
    print("\n" + "=" * 60)
    print("3. SelectKBest (f_classif / chi2)")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]

    # f_classif: 方差分析F值 (适用于连续特征, 分类目标)
    skb_f = SelectKBest(score_func=f_classif, k=10)
    X_f = skb_f.fit_transform(X, y)
    scores_f = pd.Series(skb_f.scores_, index=X.columns).sort_values(ascending=False)
    print(f"f_classif 得分前10:\n{scores_f.head(10)}")
    print(f"\n选中的特征: {skb_f.get_feature_names_out()}")

    # chi2: 卡方检验 (需要非负特征)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    skb_chi2 = SelectKBest(score_func=chi2, k=10)
    X_chi2 = skb_chi2.fit_transform(X_scaled, y)
    scores_chi2 = pd.Series(skb_chi2.scores_, index=X.columns).sort_values(ascending=False)
    print(f"\nchi2 得分前10:\n{scores_chi2.head(10)}")
    print(f"\n选中的特征: {skb_chi2.get_feature_names_out()}")

    # 对比两种方法
    f_selected = set(skb_f.get_feature_names_out())
    chi2_selected = set(skb_chi2.get_feature_names_out())
    print(f"\nf_classif 和 chi2 共同选中: {f_selected & chi2_selected}")
    print(f"仅 f_classif 选中: {f_selected - chi2_selected}")
    print(f"仅 chi2 选中: {chi2_selected - f_selected}")


def demo_mutual_info(df):
    """mutual_info_classif: 互信息法"""
    print("\n" + "=" * 60)
    print("4. mutual_info_classif 互信息法")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    print(f"互信息得分前10:\n{mi_series.head(10)}")
    print(f"\n互信息得分为0的特征数: {(mi_scores == 0).sum()}")
    print(f"互信息得分 > 0.1 的特征数: {(mi_scores > 0.1).sum()}")

    # 使用 SelectKBest + mutual_info_classif
    skb_mi = SelectKBest(score_func=mutual_info_classif, k=10)
    X_mi = skb_mi.fit_transform(X, y)
    print(f"\n互信息法选中的10个特征: {skb_mi.get_feature_names_out()}")

    # 与 f_classif 对比
    skb_f = SelectKBest(score_func=f_classif, k=10)
    skb_f.fit_transform(X, y)
    mi_set = set(skb_mi.get_feature_names_out())
    f_set = set(skb_f.get_feature_names_out())
    print(f"\n互信息 vs F值 共同选中: {mi_set & f_set}")
    print("→ 互信息能捕捉非线性关系, F值只捕捉线性关系")


def demo_rfe(df):
    """RFE: 递归特征消除"""
    print("\n" + "=" * 60)
    print("5. RFE 递归特征消除")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]

    # 使用逻辑回归作为基估计器
    estimator = LogisticRegression(max_iter=5000, random_state=42)
    rfe = RFE(estimator, n_features_to_select=10, step=1)
    X_rfe = rfe.fit_transform(X, y)

    print(f"RFE 选中特征数: {rfe.n_features_}")
    print(f"特征排名 (1=选中):\n{pd.Series(rfe.ranking_, index=X.columns).sort_values()}")
    print(f"\n选中的特征: {rfe.get_feature_names_out()}")

    # 查看被排除的特征
    not_selected = X.columns[~rfe.support_].tolist()
    print(f"\n被排除的特征 ({len(not_selected)}个): {not_selected[:5]}...")

    # step 参数: 每次迭代移除的特征数
    rfe_step5 = RFE(estimator, n_features_to_select=10, step=5)
    rfe_step5.fit_transform(X, y)
    print(f"\nstep=5 时选中特征: {rfe_step5.get_feature_names_out()}")
    print("→ step越大, 计算越快, 但可能跳过最优特征组合")


def demo_select_from_model(df):
    """SelectFromModel + RandomForest: 基于模型的特征选择"""
    print("\n" + "=" * 60)
    print("6. SelectFromModel + RandomForest 基于模型的特征选择")
    print("=" * 60)

    X = df.drop("target", axis=1)
    y = df["target"]

    # 随机森林特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"随机森林特征重要性前10:\n{importances.head(10)}")

    # SelectFromModel: 基于重要性阈值选择
    sfm_median = SelectFromModel(rf, threshold="median")
    X_sfm = sfm_median.fit_transform(X, y)
    print(f"\nthreshold='median' 选中特征: {sfm_median.get_feature_names_out()}")
    print(f"选中特征数: {X_sfm.shape[1]}")

    # threshold='mean'
    sfm_mean = SelectFromModel(rf, threshold="mean")
    X_sfm_mean = sfm_mean.fit_transform(X, y)
    print(f"\nthreshold='mean' 选中特征数: {X_sfm_mean.shape[1]}")

    # 自定义阈值
    sfm_custom = SelectFromModel(rf, threshold=0.01)
    X_sfm_custom = sfm_custom.fit_transform(X, y)
    print(f"\nthreshold=0.01 选中特征数: {X_sfm_custom.shape[1]}")
    print(f"选中特征: {sfm_custom.get_feature_names_out()}")

    # max_features 参数
    sfm_max = SelectFromModel(rf, max_features=10)
    X_sfm_max = sfm_max.fit_transform(X, y)
    print(f"\nmax_features=10 选中特征: {sfm_max.get_feature_names_out()}")


def main():
    df = load_data()

    demo_variance_threshold(df)
    demo_correlation_selection(df)
    demo_select_kbest(df)
    demo_mutual_info(df)
    demo_rfe(df)
    demo_select_from_model(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  VarianceThreshold    - 移除低方差特征, 简单快速")
    print("  相关性选择           - 移除与目标无关/特征间冗余的特征")
    print("  SelectKBest          - 基于统计检验选Top-K")
    print("    f_classif          - F方差分析, 捕捉线性关系")
    print("    chi2               - 卡方检验, 需非负特征")
    print("  mutual_info_classif  - 互信息, 捕捉非线性关系")
    print("  RFE                  - 递归消除, 计算量较大但精确")
    print("  SelectFromModel      - 基于模型重要性, 高效实用")
    print("=" * 60)


if __name__ == "__main__":
    main()
