# 数据来源: 模拟用户-物品评分数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# 加载数据
data_dir = Path(__file__).parent
df_ratings = pd.read_csv(data_dir / "ratings_data.csv")
rating_matrix = df_ratings.pivot_table(index="用户", columns="物品", values="评分")

# 划分训练集和测试集
def train_test_split_ratings(df, test_ratio=0.2):
    test_idx = np.random.choice(len(df), int(len(df) * test_ratio), replace=False)
    test = df.iloc[test_idx]
    train = df.drop(test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)

train_df, test_df = train_test_split_ratings(df_ratings)
print(f"训练集: {len(train_df)}条, 测试集: {len(test_df)}条")

# 简单的基于均值的预测器（基线）
train_matrix = train_df.pivot_table(index="用户", columns="物品", values="评分")
global_mean = train_df["评分"].mean()

# === 评分预测评估指标 ===
print("\n=== 评分预测评估指标 ===")

# 用全局均值预测
y_true = test_df["评分"].values
y_pred_mean = np.full_like(y_true, global_mean, dtype=float)

rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
mae = mean_absolute_error(y_true, y_pred_mean)
print(f"全局均值预测: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# 用用户均值预测
user_means = train_df.groupby("用户")["评分"].mean()
y_pred_user = test_df["用户"].map(user_means).fillna(global_mean).values
rmse_user = np.sqrt(mean_squared_error(y_true, y_pred_user))
mae_user = mean_absolute_error(y_true, y_pred_user)
print(f"用户均值预测: RMSE = {rmse_user:.4f}, MAE = {mae_user:.4f}")

# === 排序评估指标 ===
print("\n=== 排序评估指标 ===")

# 模拟推荐列表和真实相关物品
def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

# 构建每个用户的推荐列表（按预测评分排序）
from sklearn.metrics.pairwise import cosine_similarity

train_filled = train_matrix.fillna(0)
user_sim = cosine_similarity(train_filled)
user_sim_df = pd.DataFrame(user_sim, index=train_matrix.index, columns=train_matrix.index)

def predict_all_items(user, sim_df, mat, k=5):
    predictions = {}
    for item in mat.columns:
        if pd.isna(mat.loc[user, item]):
            neighbors = sim_df.loc[user].drop(user).nlargest(k)
            num, den = 0.0, 0.0
            for nb, sim in neighbors.items():
                if not pd.isna(mat.loc[nb, item]):
                    num += sim * mat.loc[nb, item]
                    den += abs(sim)
            predictions[item] = num / den if den != 0 else global_mean
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# 对测试集中的用户计算排序指标
k = 5
precisions, recalls, ndcgs = [], [], []
for user in test_df["用户"].unique():
    relevant = test_df[(test_df["用户"] == user) & (test_df["评分"] >= 4)]["物品"].tolist()
    if not relevant:
        continue
    recommended = [item for item, _ in predict_all_items(user, user_sim_df, train_matrix, k=5)]
    precisions.append(precision_at_k(recommended, relevant, k))
    recalls.append(recall_at_k(recommended, relevant, k))
    ndcgs.append(ndcg_at_k(recommended, relevant, k))

print(f"Precision@{k}: {np.mean(precisions):.4f}")
print(f"Recall@{k}: {np.mean(recalls):.4f}")
print(f"NDCG@{k}: {np.mean(ndcgs):.4f}")

# === 覆盖率与多样性 ===
print("\n=== 覆盖率与多样性 ===")

all_recommended = set()
for user in train_matrix.index:
    recs = predict_all_items(user, user_sim_df, train_matrix, k=5)[:5]
    all_recommended.update([item for item, _ in recs])

coverage = len(all_recommended) / len(train_matrix.columns)
print(f"物品覆盖率: {coverage:.4f} ({len(all_recommended)}/{len(train_matrix.columns)})")

# 多样性: 推荐列表中物品的平均不相似度
def intra_list_diversity(recommended, sim_matrix):
    if len(recommended) < 2:
        return 0
    total_sim = 0
    count = 0
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            if recommended[i] in sim_matrix.index and recommended[j] in sim_matrix.columns:
                total_sim += sim_matrix.loc[recommended[i], recommended[j]]
                count += 1
    return 1 - total_sim / count if count > 0 else 0

item_sim = cosine_similarity(train_filled.T)
item_sim_df = pd.DataFrame(item_sim, index=train_matrix.columns, columns=train_matrix.columns)

diversities = []
for user in train_matrix.index[:10]:
    recs = [item for item, _ in predict_all_items(user, user_sim_df, train_matrix, k=5)[:5]]
    div = intra_list_diversity(recs, item_sim_df)
    diversities.append(div)
print(f"推荐列表平均多样性(ILS): {np.mean(diversities):.4f}")
