# 依赖库最低版本要求: scikit-learn>=1.3, pandas>=2.0, numpy>=1.24, scipy>=1.10
# 数据来源: sklearn.datasets.load_iris (鸢尾花数据集)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载鸢尾花数据集并转为DataFrame"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    print("=== 原始数据概览 ===")
    print(df.head())
    print(f"\n原始数据统计描述:\n{df.describe()}")
    return df


def demo_minmax_scaler(df):
    """MinMaxScaler: 将特征缩放到 [0, 1] 范围"""
    print("\n" + "=" * 60)
    print("1. MinMaxScaler 归一化 (缩放到 [0, 1])")
    print("=" * 60)

    features = df.drop("target", axis=1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(scaled_data, columns=features.columns)
    print(f"缩放后数据:\n{scaled_df.head()}")
    print(f"\n缩放后统计描述:\n{scaled_df.describe().loc[['min', 'max', 'mean', 'std']]}")

    print(f"\nscaler.data_min_ (每列最小值): {scaler.data_min_}")
    print(f"scaler.data_max_ (每列最大值): {scaler.data_max_}")
    print(f"scaler.scale_ (缩放系数): {scaler.scale_}")
    print(f"scaler.min_ (偏移量): {scaler.min_}")

    # 自定义范围 [0, 5]
    scaler_custom = MinMaxScaler(feature_range=(0, 5))
    scaled_custom = scaler_custom.fit_transform(features)
    print(f"\n自定义范围 [0, 5] 缩放后统计:\n"
          f"min={scaled_custom.min(axis=0)}, max={scaled_custom.max(axis=0)}")

    return scaler, scaled_df


def demo_standard_scaler(df):
    """StandardScaler: 标准化 (均值=0, 标准差=1)"""
    print("\n" + "=" * 60)
    print("2. StandardScaler 标准化 (均值=0, 标准差=1)")
    print("=" * 60)

    features = df.drop("target", axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(scaled_data, columns=features.columns)
    print(f"标准化后数据:\n{scaled_df.head()}")
    print(f"\n标准化后统计描述:\n{scaled_df.describe().loc[['mean', 'std', 'min', 'max']]}")

    print(f"\nscaler.mean_ (每列均值): {scaler.mean_}")
    print(f"scaler.var_ (每列方差): {scaler.var_}")
    print(f"scaler.scale_ (每列标准差): {scaler.scale_}")
    print(f"scaler.n_samples_seen_ (样本数): {scaler.n_samples_seen_}")

    return scaler, scaled_df


def demo_maxabs_scaler(df):
    """MaxAbsScaler: 按最大绝对值缩放到 [-1, 1]"""
    print("\n" + "=" * 60)
    print("3. MaxAbsScaler 缩放 (按最大绝对值缩放到 [-1, 1])")
    print("=" * 60)

    features = df.drop("target", axis=1)
    scaler = MaxAbsScaler()
    scaled_data = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(scaled_data, columns=features.columns)
    print(f"MaxAbs缩放后数据:\n{scaled_df.head()}")
    print(f"\nMaxAbs缩放后统计描述:\n{scaled_df.describe().loc[['min', 'max', 'mean', 'std']]}")

    print(f"\nscaler.max_abs_ (每列最大绝对值): {scaler.max_abs_}")
    print(f"scaler.scale_ (缩放系数): {scaler.scale_}")

    return scaler, scaled_df


def demo_comparison(df):
    """对比三种缩放方法的效果"""
    print("\n" + "=" * 60)
    print("4. 三种缩放方法对比")
    print("=" * 60)

    features = df.drop("target", axis=1)
    col = features.columns[0]
    original = features[col].values

    minmax = MinMaxScaler().fit_transform(features[[col]])
    standard = StandardScaler().fit_transform(features[[col]])
    maxabs = MaxAbsScaler().fit_transform(features[[col]])

    comparison = pd.DataFrame({
        "原始值": original,
        "MinMaxScaler": minmax.flatten(),
        "StandardScaler": standard.flatten(),
        "MaxAbsScaler": maxabs.flatten(),
    })
    print(f"特征 '{col}' 的缩放对比:\n{comparison.head(10)}")

    stats = comparison.describe().loc[["mean", "std", "min", "max"]]
    print(f"\n各方法统计对比:\n{stats}")


def demo_inverse_transform(df):
    """inverse_transform: 将缩放后的数据还原"""
    print("\n" + "=" * 60)
    print("5. inverse_transform 逆变换 (还原数据)")
    print("=" * 60)

    features = df.drop("target", axis=1)

    # MinMaxScaler 逆变换
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    recovered = scaler.inverse_transform(scaled)

    print(f"原始数据前3行:\n{features.values[:3]}")
    print(f"\n缩放后前3行:\n{scaled[:3]}")
    print(f"\n还原后前3行:\n{recovered[:3]}")
    print(f"\n还原是否完全一致: {np.allclose(features.values, recovered)}")

    # StandardScaler 逆变换
    std_scaler = StandardScaler()
    std_scaled = std_scaler.fit_transform(features)
    std_recovered = std_scaler.inverse_transform(std_scaled)
    print(f"\nStandardScaler 还原是否完全一致: {np.allclose(features.values, std_recovered)}")


def demo_fit_vs_transform(df):
    """fit / transform / fit_transform 的区别"""
    print("\n" + "=" * 60)
    print("6. fit / transform / fit_transform 的区别")
    print("=" * 60)

    features = df.drop("target", axis=1)
    train_data = features.iloc[:100]
    test_data = features.iloc[100:]

    # 方式1: fit_transform 一步到位 (训练集)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    print(f"训练集 fit_transform 结果前3行:\n{train_scaled[:3]}")

    # 方式2: 分步操作 fit → transform (等价于 fit_transform)
    scaler2 = StandardScaler()
    scaler2.fit(train_data)
    train_scaled2 = scaler2.transform(train_data)
    print(f"\n分步 fit → transform 结果前3行:\n{train_scaled2[:3]}")
    print(f"两种方式结果是否一致: {np.allclose(train_scaled, train_scaled2)}")

    # 关键: 测试集只调用 transform (使用训练集的参数)
    test_scaled = scaler.transform(test_data)
    print(f"\n测试集 transform 结果前3行:\n{test_scaled[:3]}")
    print(f"注意: 测试集使用训练集的 mean_={scaler.mean_[:2]}... 和 scale_={scaler.scale_[:2]}...")

    # 错误做法: 对测试集调用 fit_transform
    wrong_scaler = StandardScaler()
    test_wrong = wrong_scaler.fit_transform(test_data)
    print(f"\n错误做法 (测试集 fit_transform) 前3行:\n{test_wrong[:3]}")
    print(f"正确与错误结果是否一致: {np.allclose(test_scaled, test_wrong)}")
    print("→ 不一致! 测试集不应重新 fit, 否则会造成数据泄露")


def main():
    df = load_data()

    demo_minmax_scaler(df)
    demo_standard_scaler(df)
    demo_maxabs_scaler(df)
    demo_comparison(df)
    demo_inverse_transform(df)
    demo_fit_vs_transform(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  MinMaxScaler   - 缩放到 [0,1], 适合无显著异常值、有界特征")
    print("  StandardScaler - 标准化为均值0标准差1, 适合大多数ML算法")
    print("  MaxAbsScaler   - 按最大绝对值缩放到 [-1,1], 适合稀疏数据")
    print("  fit_transform  = fit + transform, 训练集用, 测试集只用 transform")
    print("  inverse_transform 可将缩放后数据还原为原始尺度")
    print("=" * 60)


if __name__ == "__main__":
    main()
