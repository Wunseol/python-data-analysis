# 数据来源: 模拟用户-物品评分数据
# 依赖库最低版本要求: scikit-learn>=1.3, pandas>=2.0

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# 推荐系统分类说明
print("=" * 50)
print("推荐系统主要分类:")
print("  1. 协同过滤(Collaborative Filtering): 基于用户行为相似性")
print("  2. 基于内容(Content-Based): 基于物品特征相似性")
print("  3. 混合推荐(Hybrid): 结合多种方法")
print("=" * 50)

# 模拟用户和物品
n_users = 20
n_items = 15
users = [f"用户{i+1}" for i in range(n_users)]
items = [f"物品{j+1}" for j in range(n_items)]

# 模拟评分数据（稀疏矩阵，约30%缺失）
ratings_data = []
for u in range(n_users):
    for i in range(n_items):
        if np.random.random() < 0.7:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
            ratings_data.append({"用户": users[u], "物品": items[i], "评分": rating})

df_ratings = pd.DataFrame(ratings_data)
print(f"\n评分数据形状: {df_ratings.shape}")
print(f"评分数据前10行:")
print(df_ratings.head(10))

# 构建用户-物品评分矩阵
rating_matrix = df_ratings.pivot_table(index="用户", columns="物品", values="评分")
print(f"\n用户-物品矩阵形状: {rating_matrix.shape}")
print(f"矩阵稀疏度: {rating_matrix.isna().sum().sum() / (n_users * n_items):.2%}")

# 数据探索
print(f"\n--- 数据探索 ---")
print(f"评分范围: {df_ratings['评分'].min()} ~ {df_ratings['评分'].max()}")
print(f"平均评分: {df_ratings['评分'].mean():.2f}")
print(f"评分分布:\n{df_ratings['评分'].value_counts().sort_index()}")
print(f"\n每用户评分数量:\n{df_ratings.groupby('用户')['评分'].count().describe()}")
print(f"\n每物品评分数量:\n{df_ratings.groupby('物品')['评分'].count().describe()}")

# 保存模拟数据供后续案例使用
data_dir = Path(__file__).parent
df_ratings.to_csv(data_dir / "ratings_data.csv", index=False)
print(f"\n评分数据已保存至: {data_dir / 'ratings_data.csv'}")
