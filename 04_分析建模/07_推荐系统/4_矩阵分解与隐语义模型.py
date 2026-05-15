# 数据来源: 模拟用户-物品评分数据

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import NMF, TruncatedSVD

np.random.seed(42)

# 加载数据
data_dir = Path(__file__).parent
df_ratings = pd.read_csv(data_dir / "ratings_data.csv")
rating_matrix = df_ratings.pivot_table(index="用户", columns="物品", values="评分")
print(f"评分矩阵形状: {rating_matrix.shape}")

# 用均值填充缺失值
R = rating_matrix.fillna(rating_matrix.mean().mean()).values
global_mean = rating_matrix.mean().mean()

# SVD矩阵分解
print("\n=== SVD矩阵分解 ===")
svd = TruncatedSVD(n_components=10, random_state=42)
U_svd = svd.fit_transform(R)
V_svd = svd.components_
R_pred_svd = U_svd @ V_svd
print(f"U矩阵形状: {U_svd.shape}")
print(f"V矩阵形状: {V_svd.shape}")
print(f"解释方差比: {svd.explained_variance_ratio_.sum():.4f}")

# 不同隐因子维度下的解释方差
print("\n--- 隐因子维度选择 ---")
for k in [3, 5, 8, 10, 12]:
    svd_k = TruncatedSVD(n_components=k, random_state=42)
    svd_k.fit(R)
    var_ratio = svd_k.explained_variance_ratio_.sum()
    print(f"  k={k}: 解释方差比 = {var_ratio:.4f}")

# NMF非负矩阵分解
print("\n=== NMF非负矩阵分解 ===")
R_nonneg = np.clip(R, 0.5, None)
nmf = NMF(n_components=8, random_state=42, max_iter=500)
U_nmf = nmf.fit_transform(R_nonneg)
V_nmf = nmf.components_
R_pred_nmf = U_nmf @ V_nmf
print(f"U矩阵形状: {U_nmf.shape}")
print(f"V矩阵形状: {V_nmf.shape}")
print(f"重构误差: {nmf.reconstruction_err_:.4f}")

# 隐因子含义分析
print("\n--- NMF隐因子分析(前3个因子) ---")
factor_df = pd.DataFrame(V_nmf[:3], columns=rating_matrix.columns,
                         index=[f"因子{i+1}" for i in range(3)])
for i in range(3):
    top_items = factor_df.iloc[i].nlargest(3)
    print(f"  因子{i+1} 权重最高的物品: {dict(top_items.round(3))}")

# 评分预测与推荐
target_user = "用户1"
user_idx = list(rating_matrix.index).index(target_user)
unrated = rating_matrix.loc[target_user][rating_matrix.loc[target_user].isna()].index

# SVD预测
print(f"\n--- {target_user} 评分预测(SVD) ---")
preds_svd = {}
for item in unrated:
    item_idx = list(rating_matrix.columns).index(item)
    pred = R_pred_svd[user_idx, item_idx]
    preds_svd[item] = np.clip(pred, 1, 5)
    print(f"  {item}: {preds_svd[item]:.2f}")

# NMF预测
print(f"\n--- {target_user} 评分预测(NMF) ---")
preds_nmf = {}
for item in unrated:
    item_idx = list(rating_matrix.columns).index(item)
    pred = R_pred_nmf[user_idx, item_idx]
    preds_nmf[item] = np.clip(pred, 1, 5)
    print(f"  {item}: {preds_nmf[item]:.2f}")

# Top-N推荐对比
top_n = 5
print(f"\n--- Top-{top_n}推荐对比 ---")
print("SVD:", [item for item, _ in sorted(preds_svd.items(), key=lambda x: x[1], reverse=True)[:top_n]])
print("NMF:", [item for item, _ in sorted(preds_nmf.items(), key=lambda x: x[1], reverse=True)[:top_n]])
