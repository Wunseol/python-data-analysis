# 数据来源: 自建偏态分布数据

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_skewed_data():
    """构建偏态分布数据"""
    np.random.seed(42)
    n = 1000

    right_skewed = np.random.exponential(scale=50, size=n)
    left_skewed = 100 - np.random.exponential(scale=20, size=n)
    bimodal = np.concatenate([
        np.random.normal(loc=20, scale=5, size=n // 2),
        np.random.normal(loc=80, scale=5, size=n // 2),
    ])
    heavy_tailed = np.random.standard_t(df=3, size=n) * 20 + 50

    df = pd.DataFrame({
        "right_skewed": right_skewed,
        "left_skewed": left_skewed,
        "bimodal": bimodal,
        "heavy_tailed": heavy_tailed,
    })

    print("=== 偏态数据概览 ===")
    print(df.describe())
    print(f"\n偏度 (skewness):")
    for col in df.columns:
        print(f"  {col}: {df[col].skew():.3f}")
    print(f"\n峰度 (kurtosis):")
    for col in df.columns:
        print(f"  {col}: {df[col].kurtosis():.3f}")

    return df


def demo_log_transform(df):
    """对数变换: 处理右偏分布"""
    print("\n" + "=" * 60)
    print("1. 对数变换 (log transform)")
    print("=" * 60)

    col = "right_skewed"
    original = df[col]

    # np.log: 自然对数 (数据必须 > 0)
    log_transformed = np.log(original)
    print(f"原始偏度: {original.skew():.3f}")
    print(f"log 偏度: {log_transformed.skew():.3f}")

    # np.log1p: log(1+x), 对小值更稳健
    log1p_transformed = np.log1p(original)
    print(f"log1p 偏度: {log1p_transformed.skew():.3f}")

    # np.log2 / np.log10
    log10_transformed = np.log10(original)
    print(f"log10 偏度: {log10_transformed.skew():.3f}")

    # 逆变换: exp
    recovered = np.exp(log_transformed)
    print(f"\n逆变换还原是否一致: {np.allclose(original.values, recovered.values)}")

    # 对比
    comparison = pd.DataFrame({
        "原始值": original,
        "log": log_transformed,
        "log1p": log1p_transformed,
    })
    print(f"\n变换对比 (前5行):\n{comparison.head()}")

    # 对左偏数据: 先取负再log
    left_col = "left_skewed"
    left_data = df[left_col]
    left_fixed = np.log(np.max(left_data) + 1 - left_data)
    print(f"\n左偏数据: 原始偏度={left_data.skew():.3f}, "
          f"变换后偏度={pd.Series(left_fixed).skew():.3f}")


def demo_boxcox(df):
    """Box-Cox 变换 (scipy.stats.boxcox)"""
    print("\n" + "=" * 60)
    print("2. Box-Cox 变换 (scipy.stats.boxcox)")
    print("=" * 60)

    col = "right_skewed"
    data = df[col].values

    # Box-Cox 变换 (数据必须 > 0)
    transformed, lambda_val = stats.boxcox(data)
    print(f"最优 lambda: {lambda_val:.4f}")
    print(f"原始偏度: {pd.Series(data).skew():.3f}")
    print(f"Box-Cox 后偏度: {pd.Series(transformed).skew():.3f}")

    # lambda 含义
    print(f"\nlambda 含义:")
    print(f"  λ ≈ 0  → 对数变换")
    print(f"  λ = 0.5 → 平方根变换")
    print(f"  λ = 1   → 无变换")
    print(f"  λ = -1  → 取倒数变换")

    # 逆变换
    recovered = stats.inv_boxcox(transformed, lambda_val)
    print(f"\n逆变换还原是否一致: {np.allclose(data, recovered)}")

    # 对不同列应用 Box-Cox
    for c in ["right_skewed", "heavy_tailed"]:
        positive_data = df[c] - df[c].min() + 1
        bc_result, bc_lambda = stats.boxcox(positive_data)
        print(f"\n{c}: λ={bc_lambda:.4f}, "
              f"原始偏度={positive_data.skew():.3f}, "
              f"变换后偏度={pd.Series(bc_result).skew():.3f}")

    print("\n注意: Box-Cox 要求数据严格为正, 否则需先平移")


def demo_yeo_johnson(df):
    """Yeo-Johnson 变换 (PowerTransformer)"""
    print("\n" + "=" * 60)
    print("3. Yeo-Johnson 变换 (sklearn PowerTransformer)")
    print("=" * 60)

    # PowerTransformer 默认 method='yeo-johnson'
    pt = PowerTransformer(method="yeo-johnson", standardize=True)

    for col in df.columns:
        data = df[[col]].values
        transformed = pt.fit_transform(data)
        print(f"{col}:")
        print(f"  原始偏度: {df[col].skew():.3f}")
        print(f"  变换后偏度: {pd.Series(transformed.flatten()).skew():.3f}")
        print(f"  lambda: {pt.lambdas_[0]:.4f}")
        print(f"  standardize=True → 均值≈{transformed.mean():.4f}, 标准差≈{transformed.std():.4f}")

    # standardize=False: 只做幂变换, 不做标准化
    pt_no_std = PowerTransformer(method="yeo-johnson", standardize=False)
    transformed_no_std = pt_no_std.fit_transform(df[["right_skewed"]])
    print(f"\nstandardize=False: 均值={transformed_no_std.mean():.4f}, "
          f"标准差={transformed_no_std.std():.4f}")

    # Box-Cox via PowerTransformer
    pt_bc = PowerTransformer(method="box-cox", standardize=True)
    positive_data = df[["right_skewed"]]
    bc_result = pt_bc.fit_transform(positive_data)
    print(f"\nPowerTransformer(method='box-cox'): λ={pt_bc.lambdas_[0]:.4f}")

    # 逆变换
    recovered = pt.inverse_transform(pt.transform(df[["right_skewed"]]))
    print(f"\n逆变换还原是否一致: {np.allclose(df[['right_skewed']].values, recovered)}")

    print("\nYeo-Johnson vs Box-Cox:")
    print("  Yeo-Johnson: 支持正/负/零值, 更通用")
    print("  Box-Cox:     只支持正值, 但在正值数据上可能更优")


def demo_quantile_transformer(df):
    """QuantileTransformer: 分位数变换"""
    print("\n" + "=" * 60)
    print("4. QuantileTransformer 分位数变换")
    print("=" * 60)

    # output_distribution='normal': 映射到正态分布
    qt_normal = QuantileTransformer(
        n_quantiles=300,
        output_distribution="normal",
        random_state=42,
    )

    for col in df.columns:
        data = df[[col]].values
        transformed = qt_normal.fit_transform(data)
        print(f"{col}:")
        print(f"  原始偏度: {df[col].skew():.3f}, 原始峰度: {df[col].kurtosis():.3f}")
        print(f"  变换后偏度: {pd.Series(transformed.flatten()).skew():.3f}, "
              f"变换后峰度: {pd.Series(transformed.flatten()).kurtosis():.3f}")

    # output_distribution='uniform': 映射到均匀分布
    qt_uniform = QuantileTransformer(
        n_quantiles=300,
        output_distribution="uniform",
        random_state=42,
    )
    uniform_result = qt_uniform.fit_transform(df[["right_skewed"]])
    print(f"\noutput_distribution='uniform':")
    print(f"  最小值: {uniform_result.min():.4f}, 最大值: {uniform_result.max():.4f}")
    print(f"  均值: {uniform_result.mean():.4f}")

    # 逆变换
    recovered = qt_normal.inverse_transform(qt_normal.transform(df[["right_skewed"]]))
    print(f"\n逆变换还原是否一致: {np.allclose(df[['right_skewed']].values, recovered)}")

    # n_quantiles 参数
    print("\nn_quantiles 参数:")
    print("  值越大, 变换越精确, 但计算量越大")
    print("  必须 <= 样本数, 默认=1000")


def demo_before_after_comparison(df):
    """变换前后对比"""
    print("\n" + "=" * 60)
    print("5. 变换前后综合对比")
    print("=" * 60)

    col = "right_skewed"
    data = df[col]

    log_result = np.log1p(data)
    bc_result, _ = stats.boxcox(data)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    yj_result = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
    qt = QuantileTransformer(n_quantiles=300, output_distribution="normal", random_state=42)
    qt_result = qt.fit_transform(data.values.reshape(-1, 1)).flatten()

    comparison = pd.DataFrame({
        "原始": [data.skew(), data.kurtosis()],
        "log1p": [pd.Series(log_result).skew(), pd.Series(log_result).kurtosis()],
        "Box-Cox": [pd.Series(bc_result).skew(), pd.Series(bc_result).kurtosis()],
        "Yeo-Johnson": [pd.Series(yj_result).skew(), pd.Series(yj_result).kurtosis()],
        "QuantileTransformer": [pd.Series(qt_result).skew(), pd.Series(qt_result).kurtosis()],
    }, index=["偏度", "峰度"])

    print(f"特征 '{col}' 各变换方法对比:\n{comparison}")

    print("\n方法选择建议:")
    print("  log变换             - 简单, 适合中等右偏, 数据需>0")
    print("  Box-Cox            - 自适应λ, 需数据>0, scipy实现")
    print("  Yeo-Johnson        - 支持正负零值, sklearn实现, 最通用")
    print("  QuantileTransformer - 强力映射到正态/均匀, 对异常值鲁棒")


def main():
    df = build_skewed_data()

    demo_log_transform(df)
    demo_boxcox(df)
    demo_yeo_johnson(df)
    demo_quantile_transformer(df)
    demo_before_after_comparison(df)

    print("\n" + "=" * 60)
    print("总结:")
    print("  log变换              - 简单直观, 处理右偏分布")
    print("  Box-Cox             - 自适应最优λ, 数据需>0")
    print("  Yeo-Johnson         - 最通用的幂变换, 支持正负零")
    print("  QuantileTransformer - 强制映射到指定分布, 对异常值鲁棒")
    print("  偏度判断: |skew|<0.5 近似对称, 0.5~1 中等偏, >1 高度偏")
    print("=" * 60)


if __name__ == "__main__":
    main()
