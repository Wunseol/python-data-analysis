# 数据来源: seaborn diamonds 数据集 (钻石数据)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载 diamonds 数据集"""
    try:
        import seaborn as sns
        df = sns.load_dataset("diamonds")
    except ImportError:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
        df = pd.read_csv(url)

    print("=== diamonds 数据概览 ===")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n数据类型:\n{df.dtypes}")
    return df


def demo_polynomial_features(df):
    """PolynomialFeatures: 多项式特征构造"""
    print("\n" + "=" * 60)
    print("1. PolynomialFeatures 多项式特征")
    print("=" * 60)

    features = df[["carat", "depth"]].head(100)

    # 二次多项式
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    poly2_result = poly2.fit_transform(features)
    print(f"原始特征列: {features.columns.tolist()}")
    print(f"degree=2 生成特征名: {poly2.get_feature_names_out()}")
    print(f"原始形状: {features.shape} → 多项式后形状: {poly2_result.shape}")
    print(f"前3行:\n{poly2_result[:3]}")

    # 三次多项式
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    poly3_result = poly3.fit_transform(features)
    print(f"\ndegree=3 生成特征名: {poly3.get_feature_names_out()}")
    print(f"原始形状: {features.shape} → 多项式后形状: {poly3_result.shape}")

    # interaction_only=True: 只生成交互项
    poly_inter = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    inter_result = poly_inter.fit_transform(features)
    print(f"\ninteraction_only=True 特征名: {poly_inter.get_feature_names_out()}")
    print(f"形状: {inter_result.shape} (不含平方项, 只有交互项)")

    # include_bias: 是否包含常数项
    poly_bias = PolynomialFeatures(degree=2, include_bias=True)
    bias_result = poly_bias.fit_transform(features)
    print(f"\ninclude_bias=True 特征名: {poly_bias.get_feature_names_out()}")
    print(f"第一行: {bias_result[0]}")


def demo_math_transforms(df):
    """数学变换: log, sqrt, power 等"""
    print("\n" + "=" * 60)
    print("2. 数学变换 (log / sqrt / power)")
    print("=" * 60)

    price = df["price"].head(200)

    # 对数变换: 处理右偏分布
    log_price = np.log1p(price)
    print(f"原始 price 统计:\n{price.describe()}")
    print(f"\nlog1p(price) 统计:\n{log_price.describe()}")
    print(f"原始偏度: {price.skew():.2f}, 对数后偏度: {log_price.skew():.2f}")

    # 平方根变换
    sqrt_price = np.sqrt(price)
    print(f"\nsqrt(price) 偏度: {sqrt_price.skew():.2f}")

    # 幂变换
    power_02 = np.power(price, 0.2)
    print(f"price^0.2 偏度: {power_02.skew():.2f}")

    # 倒数变换
    reciprocal = 1 / (price + 1)
    print(f"1/(price+1) 偏度: {reciprocal.skew():.2f}")

    # 将变换结果加入 DataFrame
    df_transformed = pd.DataFrame({
        "price": price,
        "log_price": log_price,
        "sqrt_price": sqrt_price,
        "price^0.2": power_02,
    })
    print(f"\n变换结果对比 (前5行):\n{df_transformed.head()}")


def demo_date_features(df):
    """日期特征提取"""
    print("\n" + "=" * 60)
    print("3. 日期特征提取")
    print("=" * 60)

    # 构建日期数据
    np.random.seed(42)
    date_range = pd.date_range("2023-01-01", periods=100, freq="D")
    df_dates = pd.DataFrame({
        "date": date_range,
        "sales": np.random.poisson(lam=50, size=100),
    })

    # 提取日期特征
    df_dates["year"] = df_dates["date"].dt.year
    df_dates["month"] = df_dates["date"].dt.month
    df_dates["day"] = df_dates["date"].dt.day
    df_dates["dayofweek"] = df_dates["date"].dt.dayofweek
    df_dates["dayofyear"] = df_dates["date"].dt.dayofyear
    df_dates["weekofyear"] = df_dates["date"].dt.isocalendar().week.astype(int)
    df_dates["quarter"] = df_dates["date"].dt.quarter
    df_dates["is_weekend"] = df_dates["dayofweek"].isin([5, 6]).astype(int)
    df_dates["is_month_start"] = df_dates["date"].dt.is_month_start.astype(int)
    df_dates["is_month_end"] = df_dates["date"].dt.is_month_end.astype(int)

    print(f"日期特征提取结果:\n{df_dates.head(10)}")
    print(f"\n新增特征列: {[c for c in df_dates.columns if c not in ['date', 'sales']]}")

    # 周期性编码 (正弦/余弦变换)
    df_dates["month_sin"] = np.sin(2 * np.pi * df_dates["month"] / 12)
    df_dates["month_cos"] = np.cos(2 * np.pi * df_dates["month"] / 12)
    df_dates["dayofweek_sin"] = np.sin(2 * np.pi * df_dates["dayofweek"] / 7)
    df_dates["dayofweek_cos"] = np.cos(2 * np.pi * df_dates["dayofweek"] / 7)
    print(f"\n周期性编码 (前5行):\n{df_dates[['month', 'month_sin', 'month_cos']].head(5)}")
    print("→ 正弦/余弦编码保留周期性特征, 12月和1月在空间上相邻")


def demo_text_features():
    """文本长度特征"""
    print("\n" + "=" * 60)
    print("4. 文本长度特征")
    print("=" * 60)

    texts = pd.Series([
        "Hello world",
        "Python data analysis is fun",
        "Feature engineering",
        "Short",
        "This is a longer sentence with more words in it",
    ])

    df_text = pd.DataFrame({"text": texts})
    df_text["char_count"] = df_text["text"].str.len()
    df_text["word_count"] = df_text["text"].str.split().str.len()
    df_text["avg_word_length"] = df_text["char_count"] / df_text["word_count"]
    df_text["has_digits"] = df_text["text"].str.contains(r"\d", regex=True).astype(int)
    df_text["uppercase_ratio"] = df_text["text"].apply(
        lambda s: sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0
    )

    print(f"文本特征提取结果:\n{df_text}")


def demo_interaction_features(df):
    """交互特征构造"""
    print("\n" + "=" * 60)
    print("5. 交互特征")
    print("=" * 60)

    # 数值特征交互
    df_inter = df[["carat", "depth", "table"]].head(200).copy()

    # 两个特征相乘
    df_inter["carat_x_depth"] = df_inter["carat"] * df_inter["depth"]
    # 两个特征相除
    df_inter["carat_div_depth"] = df_inter["carat"] / df_inter["depth"]
    # 比率特征
    df_inter["depth_table_ratio"] = df_inter["depth"] / df_inter["table"]

    print(f"交互特征结果 (前5行):\n{df_inter.head()}")

    # 使用 PolynomialFeatures 生成交互项
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    inter_array = poly.fit_transform(df[["carat", "depth", "table"]].head(200))
    print(f"\nPolynomialFeatures 交互项特征名: {poly.get_feature_names_out()}")
    print(f"交互特征形状: {inter_array.shape}")


def demo_domain_features(df):
    """领域特征构造 (钻石数据特定)"""
    print("\n" + "=" * 60)
    print("6. 领域特征构造 (钻石数据)")
    print("=" * 60)

    df_domain = df.head(200).copy()

    # 体积估算 (x * y * z)
    df_domain["volume"] = df_domain["x"] * df_domain["y"] * df_domain["z"]
    # 密度估算
    df_domain["density"] = df_domain["carat"] / df_domain["volume"].replace(0, np.nan)
    # 每克拉价格
    df_domain["price_per_carat"] = df_domain["price"] / df_domain["carat"]
    # 切工比例
    df_domain["depth_ratio"] = df_domain["depth"] / df_domain["table"]
    # 尺寸比
    df_domain["length_width_ratio"] = df_domain["x"] / df_domain["y"].replace(0, np.nan)

    print(f"领域特征结果 (前5行):")
    domain_cols = ["carat", "price", "volume", "density", "price_per_carat", "depth_ratio", "length_width_ratio"]
    print(df_domain[domain_cols].head())

    print("\n领域特征构造思路:")
    print("  volume = x * y * z         → 钻石体积估算")
    print("  density = carat / volume   → 钻石密度估算")
    print("  price_per_carat            → 单位克拉价格")
    print("  depth_ratio                → 深度/台面比")
    print("  length_width_ratio         → 长宽比, 反映形状")


def main():
    df = load_data()

    demo_polynomial_features(df)
    demo_math_transforms(df)
    demo_date_features(df)
    demo_text_features()
    demo_interaction_features(df)
    demo_domain_features(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  PolynomialFeatures  - 自动生成多项式和交互特征")
    print("  数学变换           - log/sqrt/power 处理偏态分布")
    print("  日期特征           - 年/月/日/星期/季度/是否周末/周期编码")
    print("  文本特征           - 字符数/词数/平均词长/数字占比")
    print("  交互特征           - 特征乘除/比率, 捕捉联合效应")
    print("  领域特征           - 结合业务知识构造有意义的特征")
    print("=" * 60)


if __name__ == "__main__":
    main()
