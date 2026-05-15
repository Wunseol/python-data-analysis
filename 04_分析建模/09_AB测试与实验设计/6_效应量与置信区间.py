# 数据来源: 模拟数据
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)

# === Cohen's d效应量 ===
print("=" * 50)
print("1. Cohen's d效应量 — 均值差异的标准化度量")
print("=" * 50)

group_a = np.random.normal(100, 15, 200)
group_b = np.random.normal(108, 15, 200)

mean_diff = group_b.mean() - group_a.mean()
pooled_std = np.sqrt(
    ((len(group_a) - 1) * group_a.var(ddof=1) + (len(group_b) - 1) * group_b.var(ddof=1))
    / (len(group_a) + len(group_b) - 2)
)
cohens_d = mean_diff / pooled_std

print(f"对照组均值: {group_a.mean():.2f}, 标准差: {group_a.std(ddof=1):.2f}")
print(f"实验组均值: {group_b.mean():.2f}, 标准差: {group_b.std(ddof=1):.2f}")
print(f"均值差异: {mean_diff:.2f}")
print(f"合并标准差: {pooled_std:.2f}")
print(f"Cohen's d: {cohens_d:.4f}")

if abs(cohens_d) < 0.2:
    size_label = "小"
elif abs(cohens_d) < 0.5:
    size_label = "中"
elif abs(cohens_d) < 0.8:
    size_label = "大"
else:
    size_label = "非常大"
print(f"效应量大小: {size_label} (0.2小/0.5中/0.8大)")

# Hedges' g校正
n_a, n_b = len(group_a), len(group_b)
df = n_a + n_b - 2
correction = 1 - 3 / (4 * df - 1)
hedges_g = cohens_d * correction
print(f"Hedges' g(小样本校正): {hedges_g:.4f}")

# === 均值差异置信区间 ===
print("\n" + "=" * 50)
print("2. 均值差异置信区间")
print("=" * 50)

se_diff = np.sqrt(group_a.var(ddof=1) / n_a + group_b.var(ddof=1) / n_b)
t_crit = stats.t.ppf(0.975, df)

ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff
print(f"均值差异: {mean_diff:.2f}")
print(f"标准误: {se_diff:.4f}")
print(f"95%置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"结论: 区间{'不包含0' if ci_lower > 0 or ci_upper < 0 else '包含0'}，差异{'显著' if ci_lower > 0 or ci_upper < 0 else '不显著'}")

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t检验: t={t_stat:.4f}, p={p_val:.4f}")

# === 比例差异置信区间 ===
print("\n" + "=" * 50)
print("3. 比例差异置信区间")
print("=" * 50)

n1, n2 = 1000, 1000
p1_hat = 0.08
p2_hat = 0.11
diff_prop = p2_hat - p1_hat

se_prop = np.sqrt(p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2)
z_crit = stats.norm.ppf(0.975)
ci_prop_lower = diff_prop - z_crit * se_prop
ci_prop_upper = diff_prop + z_crit * se_prop

print(f"对照组转化率: {p1_hat:.2%}")
print(f"实验组转化率: {p2_hat:.2%}")
print(f"比例差异: {diff_prop:.2%}")
print(f"95%置信区间: [{ci_prop_lower:.2%}, {ci_prop_upper:.2%}]")

# Cohen's h效应量(比例)
h = 2 * np.arcsin(np.sqrt(p2_hat)) - 2 * np.arcsin(np.sqrt(p1_hat))
print(f"Cohen's h(比例效应量): {h:.4f}")

# === 效应量与样本量关系 ===
print("\n" + "=" * 50)
print("4. 效应量、样本量与统计功效关系")
print("=" * 50)

from statsmodels.stats.power import TTestIndPower

tp = TTestIndPower()
print(f"{'效应量d':>8} {'样本量/组(α=0.05,功效=0.8)':>25}")
print("-" * 35)
for d in [0.2, 0.3, 0.5, 0.8, 1.0]:
    n = tp.solve_power(effect_size=d, alpha=0.05, power=0.8, alternative='two-sided')
    print(f"{d:>8.1f} {int(np.ceil(n)):>25}")

# === 结果解读框架 ===
print("\n" + "=" * 50)
print("5. AB测试结果解读框架")
print("=" * 50)

conv_a, conv_b = 80, 110
n_a, n_b = 1000, 1000
p_a = conv_a / n_a
p_b = conv_b / n_b

z, p = proportions_ztest([conv_a, conv_b], [n_a, n_b])

h = 2 * np.arcsin(np.sqrt(p_b)) - 2 * np.arcsin(np.sqrt(p_a))

diff = p_b - p_a
se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
ci = (diff - z_crit * se, diff + z_crit * se)

print(f"统计显著性: p={p:.4f} ({'显著' if p < 0.05 else '不显著'})")
print(f"实际显著性: Cohen's h={h:.4f} ({'小' if abs(h) < 0.2 else '中' if abs(h) < 0.5 else '大'})")
print(f"置信区间: [{ci[0]:.2%}, {ci[1]:.2%}]")
print(f"业务影响: 转化率提升{diff:.2%}, 每千用户多转化{diff*1000:.1f}人")
