# 数据来源: 模拟数据
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)

# === 两比例z检验 ===
print("=" * 50)
print("1. 两比例z检验 — 转化率对比")
print("=" * 50)

# 模拟两组用户转化数据
n_a, n_b = 1200, 1200
p_a, p_b = 0.08, 0.10
conv_a = np.random.binomial(1, p_a, n_a).sum()
conv_b = np.random.binomial(1, p_b, n_b).sum()

print(f"对照组: {conv_a}/{n_a} = {conv_a/n_a:.4f}")
print(f"实验组: {conv_b}/{n_b} = {conv_b/n_b:.4f}")

z_stat, p_val = proportions_ztest([conv_a, conv_b], [n_a, n_b], alternative='two-sided')
print(f"z统计量: {z_stat:.4f}, p值: {p_val:.4f}")

z_one, p_one = proportions_ztest([conv_a, conv_b], [n_a, n_b], alternative='larger')
print(f"单侧检验(实验组>对照组): z={z_one:.4f}, p={p_one:.4f}")

# === 卡方独立性检验 ===
print("\n" + "=" * 50)
print("2. 卡方独立性检验 — 方案与转化是否关联")
print("=" * 50)

# 构建列联表: 3个方案 × 转化/未转化
observed = np.array([
    [80, 1120],   # 方案A
    [95, 1105],   # 方案B
    [110, 1090],  # 方案C
])

chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f"观测频次表:\n{observed}")
print(f"期望频次表:\n{expected.round(2)}")
print(f"卡方统计量: {chi2:.4f}")
print(f"自由度: {dof}")
print(f"p值: {p:.4f}")
print(f"结论: {'方案与转化率存在显著关联' if p < 0.05 else '方案与转化率无显著关联'}")

# === 卡方拟合优度检验 ===
print("\n" + "=" * 50)
print("3. 卡方拟合优度检验 — 实际分布是否符合预期")
print("=" * 50)

# 检验用户来源分布是否符合预期比例
observed_source = np.array([320, 280, 250, 150])
expected_ratio = np.array([0.35, 0.25, 0.25, 0.15])
expected_source = expected_ratio * observed_source.sum()

chi2_gof, p_gof = stats.chisquare(observed_source, f_exp=expected_source)
print(f"观测频次: {observed_source}")
print(f"期望频次: {expected_source.astype(int)}")
print(f"卡方统计量: {chi2_gof:.4f}")
print(f"p值: {p_gof:.4f}")
print(f"结论: {'分布与预期有显著差异' if p_gof < 0.05 else '分布与预期无显著差异'}")

# === 多组转化率对比 ===
print("\n" + "=" * 50)
print("4. 多组转化率对比")
print("=" * 50)

groups = ['方案A', '方案B', '方案C', '方案D']
conversions = [65, 82, 78, 90]
totals = [800, 800, 800, 800]

for g, c, t in zip(groups, conversions, totals):
    print(f"{g}: {c}/{t} = {c/t:.4f}")

chi2_multi, p_multi, dof_multi, _ = stats.chi2_contingency(
    np.array([[c, t-c] for c, t in zip(conversions, totals)])
)
print(f"卡方统计量: {chi2_multi:.4f}, 自由度: {dof_multi}, p值: {p_multi:.4f}")

# 两两比较
print("\n两两z检验(未校正):")
from itertools import combinations
for (i, j) in combinations(range(len(groups)), 2):
    z, p = proportions_ztest([conversions[i], conversions[j]], [totals[i], totals[j]])
    sig = "*" if p < 0.05 else ""
    print(f"  {groups[i]} vs {groups[j]}: z={z:.4f}, p={p:.4f} {sig}")
