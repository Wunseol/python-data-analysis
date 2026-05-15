# 数据来源: 模拟用户-物品评分数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# 加载数据
data_dir = Path(__file__).parent
df_ratings = pd.read_csv(data_dir / "ratings_data.csv")
rating_matrix = df_ratings.pivot_table(index="用户", columns="物品", values="评分")
print(f"评分矩阵形状: {rating_matrix.shape}")

# 物品相似度矩阵（基于用户评分向量）
item_matrix = rating_matrix.fillna(0).T
item_sim = cosine_similarity(item_matrix)
item_sim_df = pd.DataFrame(item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
print(f"\n--- 物品相似度矩阵(前5x5) ---")
print(item_sim_df.iloc[:5, :5].round(3))

# 基于物品的评分预测
# Amazon Item-CF原理: 用户u对物品i的预测评分 = Σ(相似物品j的评分 × 相似度) / Σ(|相似度|)
def predict_rating_item_cf(rating_matrix, item_sim_df, user, item, k=5):
    if not np.isnan(rating_matrix.loc[user, item]):
        return rating_matrix.loc[user, item]
    user_ratings = rating_matrix.loc[user].dropna()
    sim_items = item_sim_df[item].drop(item).sort_values(ascending=False)
    sim_items = sim_items[sim_items.index.isin(user_ratings.index)].head(k)
    if len(sim_items) == 0:
        return rating_matrix.mean().mean()
    num = np.sum(sim_items.values * user_ratings[sim_items.index].values)
    den = np.sum(np.abs(sim_items.values))
    return num / den if den != 0 else rating_matrix.mean().mean()

# 对目标用户预测未评分物品
target_user = "用户1"
unrated = rating_matrix.loc[target_user][rating_matrix.loc[target_user].isna()].index
print(f"\n--- {target_user} 未评分物品预测(Item-CF) ---")
predictions = {}
for item in unrated:
    pred = predict_rating_item_cf(rating_matrix, item_sim_df, target_user, item, k=5)
    predictions[item] = pred
    print(f"  {item}: 预测评分 = {pred:.2f}")

# Top-N推荐
top_n = 5
sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
print(f"\n--- {target_user} 的Top-{top_n}推荐(Item-CF) ---")
for item, score in sorted_preds[:top_n]:
    print(f"  {item}: 预测评分 = {score:.2f}")

# 对比User-CF和Item-CF的特点
print(f"\n--- User-CF vs Item-CF 对比 ---")
print("User-CF: 发现与当前用户相似的用户,推荐他们喜欢的物品")
print("  适合: 用户数较少、用户兴趣变化快的场景")
print("Item-CF: 发现与用户历史偏好物品相似的物品")
print("  适合: 物品数较少、物品相对稳定的场景(Amazon)")

# 计算物品相似度稳定性（物品相似度比用户相似度更稳定）
user_matrix = rating_matrix.fillna(0)
user_sim = cosine_similarity(user_matrix)
print(f"\n用户相似度标准差均值: {user_sim.std():.4f}")
print(f"物品相似度标准差均值: {item_sim.std():.4f}")
