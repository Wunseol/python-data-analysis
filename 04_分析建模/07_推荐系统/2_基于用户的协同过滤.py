# 数据来源: 模拟用户-物品评分数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

np.random.seed(42)

# 加载数据
data_dir = Path(__file__).parent
df_ratings = pd.read_csv(data_dir / "ratings_data.csv")
rating_matrix = df_ratings.pivot_table(index="用户", columns="物品", values="评分")
print(f"评分矩阵形状: {rating_matrix.shape}")

# 用户相似度计算
rating_filled = rating_matrix.fillna(0)

# 余弦相似度
cos_sim = cosine_similarity(rating_filled)
cos_sim_df = pd.DataFrame(cos_sim, index=rating_matrix.index, columns=rating_matrix.index)
print(f"\n--- 余弦相似度矩阵(前5x5) ---")
print(cos_sim_df.iloc[:5, :5].round(3))

# Pearson相关系数
def pearson_similarity(mat):
    n = mat.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i, j] = 1.0
            else:
                mask = ~mat.iloc[i].isna() & ~mat.iloc[j].isna()
                if mask.sum() >= 2:
                    sim[i, j] = pearsonr(mat.iloc[i][mask], mat.iloc[j][mask])[0]
                else:
                    sim[i, j] = 0.0
    return sim

pear_sim = pearson_similarity(rating_matrix)
pear_sim_df = pd.DataFrame(pear_sim, index=rating_matrix.index, columns=rating_matrix.index)
print(f"\n--- Pearson相似度矩阵(前5x5) ---")
print(pear_sim_df.iloc[:5, :5].round(3))

# 找最近邻用户
def find_k_nearest_neighbors(sim_df, user, k=5):
    sims = sim_df.loc[user].drop(user).sort_values(ascending=False)
    return sims.head(k)

target_user = "用户1"
k = 5
neighbors_cos = find_k_nearest_neighbors(cos_sim_df, target_user, k)
print(f"\n--- {target_user} 的Top-{k}最近邻(余弦) ---")
print(neighbors_cos.round(3))

neighbors_pear = find_k_nearest_neighbors(pear_sim_df, target_user, k)
print(f"\n--- {target_user} 的Top-{k}最近邻(Pearson) ---")
print(neighbors_pear.round(3))

# 预测评分（基于用户协同过滤）
def predict_rating_user_cf(rating_matrix, sim_df, user, item, k=5):
    if not np.isnan(rating_matrix.loc[user, item]):
        return rating_matrix.loc[user, item]
    neighbors = find_k_nearest_neighbors(sim_df, user, k)
    num, den = 0.0, 0.0
    for neighbor, sim in neighbors.items():
        if not np.isnan(rating_matrix.loc[neighbor, item]):
            num += sim * rating_matrix.loc[neighbor, item]
            den += abs(sim)
    return num / den if den != 0 else rating_matrix.mean().mean()

# 对目标用户的未评分物品预测
unrated = rating_matrix.loc[target_user][rating_matrix.loc[target_user].isna()].index
print(f"\n--- {target_user} 未评分物品预测 ---")
predictions = {}
for item in unrated:
    pred = predict_rating_user_cf(rating_matrix, pear_sim_df, target_user, item, k)
    predictions[item] = pred
    print(f"  {item}: 预测评分 = {pred:.2f}")

# Top-N推荐
top_n = 5
sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
print(f"\n--- {target_user} 的Top-{top_n}推荐 ---")
for item, score in sorted_preds[:top_n]:
    print(f"  {item}: 预测评分 = {score:.2f}")
