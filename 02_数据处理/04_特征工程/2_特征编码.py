# 数据来源: seaborn titanic 数据集 (泰坦尼克号乘客数据)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载 seaborn titanic 数据集"""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except ImportError:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        df = pd.read_csv(url)

    print("=== 原始数据概览 ===")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"\n各列数据类型:\n{df.dtypes}")
    print(f"\n分类特征唯一值:")
    for col in ["sex", "embarked", "class", "who", "embark_town"]:
        if col in df.columns:
            print(f"  {col}: {df[col].unique()}")
    return df


def demo_label_encoder(df):
    """LabelEncoder: 将类别标签编码为 0, 1, 2, ... 整数"""
    print("\n" + "=" * 60)
    print("1. LabelEncoder 标签编码")
    print("=" * 60)

    le = LabelEncoder()

    # 对 sex 列编码
    sex_encoded = le.fit_transform(df["sex"])
    print(f"原始 sex 列: {df['sex'].unique()}")
    print(f"编码后唯一值: {np.unique(sex_encoded)}")
    print(f"classes_: {le.classes_}")
    print(f"编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 对 embarked 列编码 (含缺失值需先处理)
    embarked_filled = df["embarked"].fillna("Unknown")
    embarked_encoded = le.fit_transform(embarked_filled)
    print(f"\n原始 embarked 列: {embarked_filled.unique()}")
    print(f"编码后唯一值: {np.unique(embarked_encoded)}")
    print(f"classes_: {le.classes_}")
    print(f"编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # inverse_transform 还原
    decoded = le.inverse_transform(embarked_encoded[:5])
    print(f"\n逆变换还原前5个: {decoded}")

    # 注意: LabelEncoder 适用于有序类别或目标变量, 不适合无序类别特征
    print("\n注意: LabelEncoder 引入了序关系, 对无序类别可能误导模型")


def demo_onehot_encoder(df):
    """OneHotEncoder (sklearn): 独热编码"""
    print("\n" + "=" * 60)
    print("2. OneHotEncoder 独热编码 (sklearn)")
    print("=" * 60)

    # 基本用法
    ohe = OneHotEncoder(sparse_output=False)
    sex_ohe = ohe.fit_transform(df[["sex"]])
    print(f"原始 sex 列前5行:\n{df['sex'].head().values}")
    print(f"\n独热编码结果前5行:\n{sex_ohe[:5]}")
    print(f"特征名: {ohe.get_feature_names_out()}")

    # 多列同时编码
    ohe_multi = OneHotEncoder(sparse_output=False)
    cols = ["sex", "embarked"]
    data_filled = df[cols].fillna("Unknown")
    multi_ohe = ohe_multi.fit_transform(data_filled)
    print(f"\n多列独热编码特征名: {ohe_multi.get_feature_names_out()}")
    print(f"编码后形状: {multi_ohe.shape}")
    print(f"编码结果前3行:\n{multi_ohe[:3]}")

    # drop_first: 去掉一个类别避免共线性
    ohe_drop = OneHotEncoder(sparse_output=False, drop="first")
    sex_drop = ohe_drop.fit_transform(df[["sex"]])
    print(f"\ndrop='first' 后编码结果前5行:\n{sex_drop[:5]}")
    print(f"特征名: {ohe_drop.get_feature_names_out()}")
    print("→ 去掉第一个类别, 减少冗余特征, 避免线性模型中的多重共线性")

    # handle_unknown: 处理未知类别
    ohe_ignore = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_ignore.fit(df[["sex"]])
    unknown_data = np.array([["male"], ["unknown_category"]])
    result = ohe_ignore.transform(unknown_data)
    print(f"\nhandle_unknown='ignore' 处理未知类别:")
    print(f"输入: {unknown_data.flatten()}")
    print(f"输出:\n{result}")
    print("→ 未知类别被编码为全0向量")

    # sparse_output: 稀疏输出节省内存
    ohe_sparse = OneHotEncoder(sparse_output=True)
    sparse_result = ohe_sparse.fit_transform(df[["sex", "embarked"]].fillna("Unknown"))
    print(f"\nsparse_output=True 结果类型: {type(sparse_result)}")
    print(f"稀疏矩阵形状: {sparse_result.shape}")
    print(f"非零元素数: {sparse_result.nnz}")
    print(f"稀疏率: {1 - sparse_result.nnz / (sparse_result.shape[0] * sparse_result.shape[1]):.2%}")


def demo_get_dummies(df):
    """pd.get_dummies(): pandas 的独热编码方式"""
    print("\n" + "=" * 60)
    print("3. pd.get_dummies() 独热编码 (pandas)")
    print("=" * 60)

    # 基本用法
    dummies = pd.get_dummies(df["sex"])
    print(f"get_dummies 结果前5行:\n{dummies.head()}")

    # 多列同时编码
    dummies_multi = pd.get_dummies(df[["sex", "embarked"]].fillna("Unknown"))
    print(f"\n多列 get_dummies 结果前3行:\n{dummies_multi.head(3)}")
    print(f"编码后列名: {dummies_multi.columns.tolist()}")

    # drop_first: 去掉第一个类别
    dummies_drop = pd.get_dummies(df["sex"], drop_first=True)
    print(f"\ndrop_first=True 结果前5行:\n{dummies_drop.head()}")
    print(f"列名: {dummies_drop.columns.tolist()}")

    # dtype 参数控制输出类型
    dummies_int = pd.get_dummies(df["sex"], dtype=int)
    print(f"\ndtype=int 结果前5行:\n{dummies_int.head()}")

    # 对比 sklearn OneHotEncoder vs pd.get_dummies
    print("\n--- sklearn OneHotEncoder vs pd.get_dummies 对比 ---")
    print("  OneHotEncoder: 可集成 Pipeline, 支持 handle_unknown, 适合生产环境")
    print("  get_dummies:   简便快捷, 适合探索性分析, 但无法处理未知类别")


def demo_target_encoding_concept(df):
    """目标编码 (Target Encoding) 概念说明"""
    print("\n" + "=" * 60)
    print("4. 目标编码 (Target Encoding) 概念")
    print("=" * 60)

    # 手动实现简单的目标编码
    target_means = df.groupby("embarked")["survived"].mean()
    print(f"各类别的目标均值 (survived):\n{target_means}")

    encoded = df["embarked"].map(target_means)
    print(f"\n目标编码结果前10行:\n{encoded.head(10)}")

    # 与原始标签对比
    comparison = pd.DataFrame({
        "embarked": df["embarked"],
        "survived": df["survived"],
        "target_encoded": encoded,
    })
    print(f"\n对比:\n{comparison.head(10)}")

    print("\n目标编码注意事项:")
    print("  1. 高基数类别特征的首选编码方式")
    print("  2. 容易过拟合, 需要正则化 (如加入噪声, 交叉验证编码)")
    print("  3. 生产环境推荐使用 category_encoders 库的 TargetEncoder")
    print("  4. sklearn >= 1.3 提供 TargetEncoder (from sklearn.preprocessing import TargetEncoder)")

    # sklearn TargetEncoder 示例 (如果可用)
    try:
        from sklearn.preprocessing import TargetEncoder
        te = TargetEncoder()
        embarked_filled = df[["embarked"]].fillna("Unknown")
        te_encoded = te.fit_transform(embarked_filled, df["survived"])
        print(f"\nsklearn TargetEncoder 编码结果前5行:\n{te_encoded[:5].flatten()}")
    except ImportError:
        print("\nsklearn TargetEncoder 不可用 (需要 scikit-learn >= 1.3)")


def main():
    df = load_data()

    demo_label_encoder(df)
    demo_onehot_encoder(df)
    demo_get_dummies(df)
    demo_target_encoding_concept(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  LabelEncoder      - 有序类别/目标变量, 编码为整数")
    print("  OneHotEncoder     - 无序类别, 独热编码, 可集成 Pipeline")
    print("  pd.get_dummies()  - 快捷独热编码, 适合探索分析")
    print("  drop_first        - 去掉一个类别避免共线性")
    print("  handle_unknown    - 处理测试集中的未知类别")
    print("  sparse_output     - 稀疏输出节省高维内存")
    print("  Target Encoding   - 高基数类别, 用目标均值编码, 需防过拟合")
    print("=" * 60)


if __name__ == "__main__":
    main()
