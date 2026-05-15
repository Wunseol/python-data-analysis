# 数据来源: 自建 DataFrame 演示数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_data():
    """构建演示数据"""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, 100),
        "income": np.random.exponential(scale=5000, size=100).round(2),
        "score": np.random.normal(loc=70, scale=15, size=100).round(1),
    })
    print("=== 演示数据概览 ===")
    print(df.head(10))
    print(f"\n统计描述:\n{df.describe()}")
    return df


def demo_pd_cut_equal_width(df):
    """pd.cut() 等宽分箱"""
    print("\n" + "=" * 60)
    print("1. pd.cut() 等宽分箱")
    print("=" * 60)

    # 指定箱数
    bins_5 = pd.cut(df["age"], bins=5)
    print(f"等宽分箱 (bins=5):\n{bins_5.head(10)}")
    print(f"\n各箱计数:\n{bins_5.value_counts().sort_index()}")

    # 自定义分箱边界
    custom_bins = [0, 25, 40, 60, 100]
    bins_custom = pd.cut(df["age"], bins=custom_bins)
    print(f"\n自定义边界分箱:\n{bins_custom.head(10)}")
    print(f"\n各箱计数:\n{bins_custom.value_counts().sort_index()}")

    # 添加自定义标签
    labels = ["青年", "中年", "中老年", "老年"]
    bins_labeled = pd.cut(df["age"], bins=custom_bins, labels=labels)
    print(f"\n带标签分箱:\n{bins_labeled.head(10)}")
    print(f"\n各箱计数:\n{bins_labeled.value_counts().sort_index()}")

    # right=False: 左闭右开区间
    bins_left = pd.cut(df["age"], bins=custom_bins, right=False)
    print(f"\nright=False (左闭右开):\n{bins_left.head(10)}")
    print(f"\n各箱计数:\n{bins_left.value_counts().sort_index()}")

    # include_lowest=True: 包含最小值
    bins_inc = pd.cut(df["age"], bins=5, include_lowest=True)
    print(f"\ninclude_lowest=True:\n{bins_inc.head(5)}")

    # retbins=True: 同时返回分箱边界
    result, bin_edges = pd.cut(df["age"], bins=5, retbins=True)
    print(f"\n分箱边界: {bin_edges}")


def demo_pd_qcut(df):
    """pd.qcut() 等频分箱 (每箱样本数大致相同)"""
    print("\n" + "=" * 60)
    print("2. pd.qcut() 等频分箱")
    print("=" * 60)

    # 4分位
    qcut_4 = pd.qcut(df["income"], q=4)
    print(f"等频分箱 (q=4):\n{qcut_4.head(10)}")
    print(f"\n各箱计数:\n{qcut_4.value_counts().sort_index()}")

    # 十分位
    qcut_10 = pd.qcut(df["score"], q=10, duplicates="drop")
    print(f"\n等频分箱 (q=10, duplicates='drop'):\n{qcut_10.head(5)}")
    print(f"\n各箱计数:\n{qcut_10.value_counts().sort_index()}")

    # 自定义分位数
    custom_q = [0, 0.2, 0.5, 0.8, 1.0]
    qcut_custom = pd.qcut(df["income"], q=custom_q)
    print(f"\n自定义分位数 {custom_q}:")
    print(f"各箱计数:\n{qcut_custom.value_counts().sort_index()}")

    # 带标签
    labels = ["低", "中低", "中高", "高"]
    qcut_labeled = pd.qcut(df["income"], q=4, labels=labels)
    print(f"\n带标签等频分箱:\n{qcut_labeled.head(10)}")
    print(f"\n各箱计数:\n{qcut_labeled.value_counts()}")

    # 等宽 vs 等频 对比
    print("\n--- 等宽 vs 等频 对比 (income 列) ---")
    cut_result = pd.cut(df["income"], bins=4)
    qcut_result = pd.qcut(df["income"], q=4)
    print(f"等宽分箱各箱计数:\n{cut_result.value_counts().sort_index()}")
    print(f"\n等频分箱各箱计数:\n{qcut_result.value_counts().sort_index()}")
    print("→ 等宽: 区间等长, 计数可能不均; 等频: 计数均匀, 区间不等长")


def demo_kbins_discretizer(df):
    """KBinsDiscretizer: sklearn 分箱离散化"""
    print("\n" + "=" * 60)
    print("3. KBinsDiscretizer 分箱离散化 (sklearn)")
    print("=" * 60)

    # uniform 策略: 等宽
    kb_uniform = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
    age_uniform = kb_uniform.fit_transform(df[["age"]])
    print(f"uniform 策略 (等宽):")
    print(f"  分箱边界: {kb_uniform.bin_edges_[0]}")
    print(f"  编码结果前10: {age_uniform[:10].flatten()}")
    print(f"  各箱计数: {pd.Series(age_uniform.flatten()).value_counts().sort_index().tolist()}")

    # quantile 策略: 等频
    kb_quantile = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    age_quantile = kb_quantile.fit_transform(df[["age"]])
    print(f"\nquantile 策略 (等频):")
    print(f"  分箱边界: {kb_quantile.bin_edges_[0]}")
    print(f"  编码结果前10: {age_quantile[:10].flatten()}")
    print(f"  各箱计数: {pd.Series(age_quantile.flatten()).value_counts().sort_index().tolist()}")

    # kmeans 策略: K均值聚类分箱
    kb_kmeans = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="kmeans")
    age_kmeans = kb_kmeans.fit_transform(df[["age"]])
    print(f"\nkmeans 策略 (K均值):")
    print(f"  分箱边界: {kb_kmeans.bin_edges_[0]}")
    print(f"  编码结果前10: {age_kmeans[:10].flatten()}")
    print(f"  各箱计数: {pd.Series(age_kmeans.flatten()).value_counts().sort_index().tolist()}")

    # encode='onehot': 输出独热编码
    kb_ohe = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="uniform")
    age_ohe = kb_ohe.fit_transform(df[["age"]])
    print(f"\nencode='onehot-dense' 结果形状: {age_ohe.shape}")
    print(f"前3行:\n{age_ohe[:3]}")

    # 多列同时分箱
    kb_multi = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    multi_binned = kb_multi.fit_transform(df[["age", "income", "score"]])
    print(f"\n多列同时分箱结果前5行:\n{multi_binned[:5]}")
    for i, col in enumerate(["age", "income", "score"]):
        print(f"  {col} 分箱边界: {kb_multi.bin_edges_[i]}")


def demo_methods_comparison(df):
    """对比不同分箱方法"""
    print("\n" + "=" * 60)
    print("4. 分箱方法综合对比")
    print("=" * 60)

    col = "income"
    data = df[[col]].values

    # pd.cut 等宽
    cut_4 = pd.cut(df[col], bins=4)
    # pd.qcut 等频
    qcut_4 = pd.qcut(df[col], q=4, duplicates="drop")
    # KBinsDiscretizer uniform
    kb_u = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
    kb_u_result = kb_u.fit_transform(data).flatten()
    # KBinsDiscretizer quantile
    kb_q = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    kb_q_result = kb_q.fit_transform(data).flatten()
    # KBinsDiscretizer kmeans
    kb_k = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="kmeans")
    kb_k_result = kb_k.fit_transform(data).flatten()

    comparison = pd.DataFrame({
        "原始值": df[col],
        "pd.cut(等宽)": cut_4.cat.codes,
        "pd.qcut(等频)": qcut_4.cat.codes,
        "KBins_uniform": kb_u_result.astype(int),
        "KBins_quantile": kb_q_result.astype(int),
        "KBins_kmeans": kb_k_result.astype(int),
    })
    print(f"各方法分箱结果对比 (前10行):\n{comparison.head(10)}")

    print("\n各方法各箱计数:")
    for method in ["pd.cut(等宽)", "pd.qcut(等频)", "KBins_uniform", "KBins_quantile", "KBins_kmeans"]:
        counts = comparison[method].value_counts().sort_index()
        print(f"  {method}: {counts.tolist()}")

    print("\n方法选择建议:")
    print("  等宽 (uniform) - 数据分布均匀时适用, 简单直观")
    print("  等频 (quantile) - 数据偏态时适用, 每箱样本均衡")
    print("  K均值 (kmeans) - 数据有自然聚类时适用, 边界更合理")
    print("  自定义边界     - 有业务含义的分箱点时首选")


def main():
    df = build_data()

    demo_pd_cut_equal_width(df)
    demo_pd_qcut(df)
    demo_kbins_discretizer(df)
    demo_methods_comparison(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  pd.cut()            - 等宽/自定义边界分箱, 简单灵活")
    print("  pd.qcut()           - 等频分箱, 每箱样本数均衡")
    print("  KBinsDiscretizer    - sklearn分箱, 可集成Pipeline")
    print("    strategy=uniform  - 等宽")
    print("    strategy=quantile - 等频")
    print("    strategy=kmeans   - K均值聚类")
    print("  encode=ordinal/onehot/onehot-dense 控制输出格式")
    print("=" * 60)


if __name__ == "__main__":
    main()
