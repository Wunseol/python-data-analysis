# 数据来源: 模拟物品特征数据与用户评分数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)

# 模拟物品特征数据
items = [f"物品{i+1}" for i in range(15)]
genres = ["动作", "喜剧", "爱情", "科幻", "恐怖", "动画", "悬疑", "纪录片",
         "冒险", "奇幻", "战争", "音乐", "历史", "体育", "犯罪"]
descriptions = {
    "物品1": "动作 科幻 冒险 未来世界 英雄拯救",
    "物品2": "喜剧 爱情 都市 浪漫邂逅",
    "物品3": "恐怖 悬疑 惊悚 午夜 诡宅",
    "物品4": "动画 奇幻 冒险 魔法世界",
    "物品5": "科幻 太空 探索 未来科技",
    "物品6": "动作 战争 历史 二战",
    "物品7": "喜剧 音乐 青春 校园",
    "物品8": "纪录片 历史 文化 传承",
    "物品9": "悬疑 犯罪 侦探 推理",
    "物品10": "爱情 奇幻 穿越 时空",
    "物品11": "动作 犯罪 警匪 追逐",
    "物品12": "科幻 人工智能 未来 反乌托邦",
    "物品13": "动画 冒险 友情 成长",
    "物品14": "恐怖 悬疑 心理 惊悚",
    "物品15": "喜剧 体育 励志 拼搏",
}

df_items = pd.DataFrame({"物品": items, "描述": [descriptions[i] for i in items]})
print("--- 物品特征数据 ---")
print(df_items)

# TF-IDF内容向量
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_items["描述"])
print(f"\nTF-IDF矩阵形状: {tfidf_matrix.shape}")
print(f"特征词: {tfidf.get_feature_names_out()}")

# 物品内容相似度
content_sim = cosine_similarity(tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim, index=items, columns=items)
print(f"\n--- 物品内容相似度(前5x5) ---")
print(content_sim_df.iloc[:5, :5].round(3))

# 基于内容的推荐
def content_based_recommend(user_history, content_sim_df, top_n=5):
    sim_scores = np.zeros(len(content_sim_df))
    for item in user_history:
        sim_scores += content_sim_df[item].values
    sim_scores /= len(user_history)
    rec_items = pd.Series(sim_scores, index=content_sim_df.columns)
    rec_items = rec_items.drop(user_history).sort_values(ascending=False)
    return rec_items.head(top_n)

# 模拟用户历史偏好
data_dir = Path(__file__).parent
df_ratings = pd.read_csv(data_dir / "ratings_data.csv")
target_user = "用户1"
user_history = df_ratings[df_ratings["用户"] == target_user].sort_values("评分", ascending=False)
liked_items = user_history[user_history["评分"] >= 4]["物品"].tolist()
print(f"\n--- {target_user} 喜欢的物品 ---")
print(liked_items)

recommendations = content_based_recommend(liked_items, content_sim_df, top_n=5)
print(f"\n--- {target_user} 基于内容的Top-5推荐 ---")
for item, score in recommendations.items():
    print(f"  {item}: 相似度 = {score:.3f}")

# 冷启动问题分析
print("\n--- 冷启动问题分析 ---")
print("新用户冷启动: 无历史行为,无法构建用户画像")
print("  解决方案: 热门推荐→用户注册标签→引导评分")
print("新物品冷启动: 无用户评分,协同过滤无法推荐")
print("  解决方案: 基于内容推荐可缓解(只要有物品特征)")

# 对比: 新物品在协同过滤vs内容推荐中的表现
new_item_desc = "动作 科幻 太空 冒险 未来"
new_item_vec = tfidf.transform([new_item_desc])
new_item_sim = cosine_similarity(new_item_vec, tfidf_matrix).flatten()
print(f"\n新物品'动作科幻太空冒险'与已有物品的内容相似度:")
for idx in np.argsort(new_item_sim)[::-1][:5]:
    print(f"  {items[idx]}: {new_item_sim[idx]:.3f}")
