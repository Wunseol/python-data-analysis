# 数据来源: 模拟MovieLens风格电影评分数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)

# 模拟MovieLens数据
n_users = 50
n_movies = 30
users = [f"用户{i+1}" for i in range(n_users)]

movie_info = {
    "电影1": {"类型": "动作 科幻 冒险", "年份": 2020},
    "电影2": {"类型": "喜剧 爱情 都市", "年份": 2021},
    "电影3": {"类型": "恐怖 悬疑 惊悚", "年份": 2019},
    "电影4": {"类型": "动画 奇幻 冒险", "年份": 2022},
    "电影5": {"类型": "科幻 太空 探索", "年份": 2020},
    "电影6": {"类型": "动作 战争 历史", "年份": 2018},
    "电影7": {"类型": "喜剧 音乐 青春", "年份": 2021},
    "电影8": {"类型": "纪录片 自然 探索", "年份": 2022},
    "电影9": {"类型": "悬疑 犯罪 侦探", "年份": 2020},
    "电影10": {"类型": "爱情 奇幻 穿越", "年份": 2019},
    "电影11": {"类型": "动作 犯罪 警匪", "年份": 2021},
    "电影12": {"类型": "科幻 人工智能 未来", "年份": 2022},
    "电影13": {"类型": "动画 冒险 友情", "年份": 2020},
    "电影14": {"类型": "恐怖 心理 惊悚", "年份": 2021},
    "电影15": {"类型": "喜剧 体育 励志", "年份": 2019},
    "电影16": {"类型": "动作 科幻 超级英雄", "年份": 2022},
    "电影17": {"类型": "爱情 文艺 治愈", "年份": 2020},
    "电影18": {"类型": "悬疑 推理 密室", "年份": 2021},
    "电影19": {"类型": "动画 童话 奇幻", "年份": 2018},
    "电影20": {"类型": "纪录片 历史 文化", "年份": 2022},
    "电影21": {"类型": "动作 间谍 惊险", "年份": 2020},
    "电影22": {"类型": "喜剧 家庭 温馨", "年份": 2021},
    "电影23": {"类型": "科幻 末日 生存", "年份": 2019},
    "电影24": {"类型": "恐怖 丧尸 逃生", "年份": 2022},
    "电影25": {"类型": "爱情 青春 校园", "年份": 2020},
    "电影26": {"类型": "冒险 探险 宝藏", "年份": 2021},
    "电影27": {"类型": "犯罪 黑帮 复仇", "年份": 2019},
    "电影28": {"类型": "喜剧 职场 奋斗", "年份": 2022},
    "电影29": {"类型": "科幻 时间旅行 悖论", "年份": 2020},
    "电影30": {"类型": "动画 环保 自然", "年份": 2021},
}
movies = list(movie_info.keys())

# 生成评分（带用户偏好倾向）
genre_prefs = np.random.dirichlet(np.ones(6), size=n_users)
genre_map = {"动作": 0, "喜剧": 1, "爱情": 2, "科幻": 3, "恐怖": 4, "动画": 5}

ratings_data = []
for u_idx, user in enumerate(users):
    for movie, info in movie_info.items():
        if np.random.random() < 0.5:
            pref_score = 0
            for genre, g_idx in genre_map.items():
                if genre in info["类型"]:
                    pref_score += genre_prefs[u_idx, g_idx]
            base = 2.5 + pref_score * 3
            rating = np.clip(base + np.random.normal(0, 0.8), 1, 5)
            ratings_data.append({"用户": user, "电影": movie, "评分": round(rating, 1)})

df = pd.DataFrame(ratings_data)
print(f"评分数据: {len(df)}条")
print(f"用户数: {df['用户'].nunique()}, 电影数: {df['电影'].nunique()}")
print(f"评分分布:\n{df['评分'].describe()}")

# 划分训练集和测试集
test_idx = np.random.choice(len(df), int(len(df) * 0.2), replace=False)
train_df = df.drop(test_idx).reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)
print(f"\n训练集: {len(train_df)}, 测试集: {len(test_df)}")

# 构建评分矩阵
train_matrix = train_df.pivot_table(index="用户", columns="电影", values="评分")
global_mean = train_df["评分"].mean()

# === 算法1: 基于用户的协同过滤 ===
print("\n" + "=" * 50)
print("算法1: 基于用户的协同过滤(User-CF)")
train_filled = train_matrix.fillna(0)
user_sim = cosine_similarity(train_filled)
user_sim_df = pd.DataFrame(user_sim, index=train_matrix.index, columns=train_matrix.index)

def predict_user_cf(user, item, sim_df, mat, k=10):
    neighbors = sim_df.loc[user].drop(user).nlargest(k)
    num, den = 0.0, 0.0
    for nb, sim in neighbors.items():
        if not pd.isna(mat.loc[nb, item]):
            num += sim * (mat.loc[nb, item] - mat.loc[nb].mean())
            den += abs(sim)
    if den == 0:
        return global_mean
    return np.clip(mat.loc[user].mean() + num / den, 1, 5)

# === 算法2: 基于物品的协同过滤 ===
print("算法2: 基于物品的协同过滤(Item-CF)")
item_sim = cosine_similarity(train_filled.T)
item_sim_df = pd.DataFrame(item_sim, index=train_matrix.columns, columns=train_matrix.columns)

def predict_item_cf(user, item, sim_df, mat, k=10):
    user_ratings = mat.loc[user].dropna()
    sim_items = sim_df[item].drop(item).sort_values(ascending=False)
    sim_items = sim_items[sim_items.index.isin(user_ratings.index)].head(k)
    if len(sim_items) == 0:
        return global_mean
    num = np.sum(sim_items.values * user_ratings[sim_items.index].values)
    den = np.sum(np.abs(sim_items.values))
    return np.clip(num / den if den != 0 else global_mean, 1, 5)

# === 算法3: NMF矩阵分解 ===
print("算法3: NMF矩阵分解")
R = train_matrix.fillna(global_mean).values
R = np.clip(R, 0.5, None)
nmf = NMF(n_components=10, random_state=42, max_iter=500)
U_nmf = nmf.fit_transform(R)
V_nmf = nmf.components_

def predict_nmf(user, item):
    u_idx = list(train_matrix.index).index(user)
    i_idx = list(train_matrix.columns).index(item)
    return np.clip(U_nmf[u_idx] @ V_nmf[:, i_idx], 1, 5)

# === 算法4: 基于内容推荐 ===
print("算法4: 基于内容推荐")
movie_descs = [movie_info[m]["类型"] for m in train_matrix.columns]
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movie_descs)
content_sim = cosine_similarity(tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim, index=train_matrix.columns, columns=train_matrix.columns)

def predict_content(user, item, mat, sim_df, k=10):
    user_ratings = mat.loc[user].dropna()
    sim_items = sim_df[item].drop(item).sort_values(ascending=False)
    sim_items = sim_items[sim_items.index.isin(user_ratings.index)].head(k)
    if len(sim_items) == 0:
        return global_mean
    num = np.sum(sim_items.values * user_ratings[sim_items.index].values)
    den = np.sum(np.abs(sim_items.values))
    return np.clip(num / den if den != 0 else global_mean, 1, 5)

# === 评估 ===
print("\n" + "=" * 50)
print("模型评估(RMSE / MAE)")
algorithms = {
    "User-CF": lambda u, i: predict_user_cf(u, i, user_sim_df, train_matrix),
    "Item-CF": lambda u, i: predict_item_cf(u, i, item_sim_df, train_matrix),
    "NMF": lambda u, i: predict_nmf(u, i),
    "Content": lambda u, i: predict_content(u, i, train_matrix, content_sim_df),
}

results = {}
for name, pred_fn in algorithms.items():
    preds, trues = [], []
    for _, row in test_df.iterrows():
        u, i = row["用户"], row["电影"]
        if u in train_matrix.index and i in train_matrix.columns:
            preds.append(pred_fn(u, i))
            trues.append(row["评分"])
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    results[name] = {"RMSE": rmse, "MAE": mae}
    print(f"  {name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# === 推荐结果展示 ===
print("\n" + "=" * 50)
target = "用户1"
unrated = train_matrix.loc[target][train_matrix.loc[target].isna()].index
print(f"为 {target} 生成推荐:")

for name, pred_fn in algorithms.items():
    recs = [(item, pred_fn(target, item)) for item in unrated]
    recs.sort(key=lambda x: x[1], reverse=True)
    top5 = recs[:5]
    print(f"\n  {name} Top-5推荐:")
    for item, score in top5:
        print(f"    {item}({movie_info[item]['类型']}): {score:.2f}")

# 清理临时数据文件
data_dir = Path(__file__).parent
temp_file = data_dir / "ratings_data.csv"
if temp_file.exists():
    temp_file.unlink()
    print(f"\n已清理临时数据文件: {temp_file}")
