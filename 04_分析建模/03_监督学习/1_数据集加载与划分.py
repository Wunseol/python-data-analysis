# 数据来源: sklearn 内置数据集 (load_iris, load_wine, fetch_california_housing) 及合成数据生成器
# 依赖库最低版本要求: scikit-learn>=1.3, numpy>=1.24, pandas>=2.0, matplotlib>=3.7
# [已弃用] load_boston 在 sklearn 1.2 中已移除，请使用 fetch_california_housing 替代

"""
Scikit-learn 数据集加载与划分
============================
本案例演示:
1. sklearn 内置数据集的加载 (分类/回归)
2. 合成数据集的生成 (make_classification, make_blobs)
3. train_test_split 数据集划分
4. stratify 分层抽样保证类别比例
5. random_state 可复现性
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split

# ============================================================
# 一、加载 sklearn 内置分类数据集
# ============================================================

# 1. 鸢尾花数据集 — 经典多分类数据集
iris = load_iris()
print("=" * 60)
print("【鸢尾花数据集 load_iris】")
print(f"特征矩阵形状: {iris.data.shape}")
print(f"目标向量形状: {iris.target.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")
print(f"类别分布: {np.bincount(iris.target)}")
print(f"前5行特征:\n{iris.data[:5]}")
print(f"前5个标签: {iris.target[:5]}")

# 2. 红酒数据集 — 13个化学特征, 3类红酒
wine = load_wine()
print("\n" + "=" * 60)
print("【红酒数据集 load_wine】")
print(f"特征矩阵形状: {wine.data.shape}")
print(f"目标向量形状: {wine.target.shape}")
print(f"特征名称: {wine.feature_names}")
print(f"类别名称: {wine.target_names}")
print(f"类别分布: {np.bincount(wine.target)}")

# 3. 加州房价数据集 — 回归数据集
# [已弃用] load_boston 在 sklearn 1.2 中已移除，请使用 fetch_california_housing 替代
california = fetch_california_housing()
print("\n" + "=" * 60)
print("【加州房价数据集 fetch_california_housing】")
print(f"特征矩阵形状: {california.data.shape}")
print(f"目标向量形状: {california.target.shape}")
print(f"特征名称: {california.feature_names}")
print(f"目标值范围: [{california.target.min():.2f}, {california.target.max():.2f}]")
print(f"目标值均值: {california.target.mean():.2f}")

# 将数据集转为 DataFrame 方便查看
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(f"\n鸢尾花数据集 DataFrame 概览:")
print(iris_df.describe())

# ============================================================
# 二、生成合成数据集
# ============================================================

# 1. make_classification — 生成分类数据
X_clf, y_clf = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)
print("\n" + "=" * 60)
print("【合成分类数据 make_classification】")
print(f"特征矩阵形状: {X_clf.shape}")
print(f"目标向量形状: {y_clf.shape}")
print(f"类别分布: {np.bincount(y_clf)}")
print(f"类别数: {len(np.unique(y_clf))}")

# 2. make_blobs — 生成聚类数据 (各向同性高斯团)
X_blob, y_blob = make_blobs(
    n_samples=300,
    n_features=2,
    centers=4,
    cluster_std=1.5,
    random_state=42
)
print("\n" + "=" * 60)
print("【合成聚类数据 make_blobs】")
print(f"特征矩阵形状: {X_blob.shape}")
print(f"目标向量形状: {y_blob.shape}")
print(f"簇数: {len(np.unique(y_blob))}")
print(f"簇中心 (前2个):\n{X_blob[:2]}")

# ============================================================
# 三、train_test_split 数据集划分
# ============================================================

# 基本划分 — 鸢尾花数据集
X_iris = iris.data
y_iris = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris,
    test_size=0.3,
    random_state=42
)
print("\n" + "=" * 60)
print("【基本划分 train_test_split】")
print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print(f"训练集标签形状: {y_train.shape}")
print(f"测试集标签形状: {y_test.shape}")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}")

# ============================================================
# 四、stratify 分层抽样
# ============================================================

# 不使用 stratify — 类别比例可能不均衡
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(
    X_iris, y_iris,
    test_size=0.3,
    random_state=42
)

# 使用 stratify — 保证训练集和测试集的类别比例与原始数据一致
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_iris, y_iris,
    test_size=0.3,
    stratify=y_iris,
    random_state=42
)

print("\n" + "=" * 60)
print("【stratify 分层抽样对比】")
print(f"原始数据类别比例: {np.bincount(y_iris) / len(y_iris)}")
print(f"无stratify训练集比例: {np.bincount(y_train_ns) / len(y_train_ns)}")
print(f"有stratify训练集比例: {np.bincount(y_train_s) / len(y_train_s)}")
print(f"无stratify测试集比例: {np.bincount(y_test_ns) / len(y_test_ns)}")
print(f"有stratify测试集比例: {np.bincount(y_test_s) / len(y_test_s)}")

# ============================================================
# 五、random_state 可复现性
# ============================================================

# 相同 random_state 得到相同划分
X_train1, X_test1, _, _ = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
X_train2, X_test2, _, _ = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
print("\n" + "=" * 60)
print("【random_state 可复现性】")
print(f"random_state=42 两次划分是否一致: {np.array_equal(X_train1, X_train2)}")

# 不同 random_state 得到不同划分
X_train3, X_test3, _, _ = train_test_split(X_iris, y_iris, test_size=0.3, random_state=0)
print(f"random_state=42 vs 0 是否一致: {np.array_equal(X_train1, X_train3)}")

# ============================================================
# 六、回归数据集划分示例
# ============================================================

X_cal = california.data
y_cal = california.target

X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(
    X_cal, y_cal,
    test_size=0.2,
    random_state=42
)
print("\n" + "=" * 60)
print("【加州房价数据集划分】")
print(f"训练集形状: {X_train_cal.shape}")
print(f"测试集形状: {X_test_cal.shape}")
print(f"训练集目标均值: {y_train_cal.mean():.4f}")
print(f"测试集目标均值: {y_test_cal.mean():.4f}")

print("\n" + "=" * 60)
print("数据集加载与划分演示完成!")
