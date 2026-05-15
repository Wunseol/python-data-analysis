# 数据来源: 模拟数据
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

np.random.seed(42)

# === 1. 模拟网页AB测试数据 ===
print("=" * 60)
print("综合案例: 网页优化AB测试")
print("=" * 60)

print("\n--- 步骤1: 实验设计 ---")
print("背景: 电商网站优化商品详情页按钮颜色")
print("指标: 购买转化率")
print("H0: 新版转化率 = 旧版转化率")
print("H1: 新版转化率 > 旧版转化率")

baseline_rate = 0.08
mde = 0.02
alpha = 0.05
power = 0.80

effect_size = proportion_effectsize(baseline_rate + mde, baseline_rate)
n_required = NormalIndPower().solve_power(
    effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided'
)
n_required = int(np.ceil(n_required))
print(f"基准转化率: {baseline_rate:.0%}")
print(f"最小可检测效应: {mde:.0%}")
print(f"每组所需样本量: {n_required}")

# === 2. 收集模拟数据 ===
print("\n--- 步骤2: 数据收集 ---")
n_a = n_required
n_b = n_required
true_rate_a = 0.08
true_rate_b = 0.098  # 新版真实转化率

conversions_a = np.random.binomial(1, true_rate_a, n_a)
conversions_b = np.random.binomial(1, true_rate_b, n_b)

conv_a = conversions_a.sum()
conv_b = conversions_b.sum()
rate_a = conv_a / n_a
rate_b = conv_b / n_b

print(f"对照组: {conv_a}/{n_a} = {rate_a:.4f}")
print(f"实验组: {conv_b}/{n_b} = {rate_b:.4f}")
print(f"观测差异: {rate_b - rate_a:.4f} ({(rate_b - rate_a)/rate_a:.2%}提升)")

# === 3. 统计检验 ===
print("\n--- 步骤3: 统计检验 ---")

# 两比例z检验
z_stat, p_value = proportions_ztest(
    [conv_a, conv_b], [n_a, n_b], alternative='two-sided'
)
print(f"两比例z检验: z={z_stat:.4f}, p={p_value:.4f}")

# 卡方检验
chi2, p_chi2, dof, expected = stats.chi2_contingency(
    np.array([[conv_a, n_a - conv_a], [conv_b, n_b - conv_b]])
)
print(f"卡方检验: χ²={chi2:.4f}, p={p_chi2:.4f}")

# Fisher精确检验(小样本时)
_, p_fisher = stats.fisher_exact(
    np.array([[conv_a, n_a - conv_a], [conv_b, n_b - conv_b]])
)
print(f"Fisher精确检验: p={p_fisher:.4f}")

# === 4. 效应量计算 ===
print("\n--- 步骤4: 效应量 ---")

h = 2 * np.arcsin(np.sqrt(rate_b)) - 2 * np.arcsin(np.sqrt(rate_a))
print(f"Cohen's h: {h:.4f}")

if abs(h) < 0.2:
    effect_label = "小"
elif abs(h) < 0.5:
    effect_label = "中"
else:
    effect_label = "大"
print(f"效应量大小: {effect_label}")

# === 5. 置信区间 ===
print("\n--- 步骤5: 置信区间 ---")

diff = rate_b - rate_a
se = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
z_crit = stats.norm.ppf(0.975)
ci_lower = diff - z_crit * se
ci_upper = diff + z_crit * se

print(f"转化率差异: {diff:.4f}")
print(f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"相对提升: {diff/rate_a:.2%}")
print(f"95%CI相对提升: [{ci_lower/rate_a:.2%}, {ci_upper/rate_a:.2%}]")

# === 6. 样本量验证 ===
print("\n--- 步骤6: 事后功效验证 ---")

observed_effect = proportion_effectsize(rate_b, rate_a)
post_hoc_power = NormalIndPower().power(
    effect_size=observed_effect, nobs1=n_a, alpha=alpha, alternative='two-sided'
)
print(f"观测效应量: {observed_effect:.4f}")
print(f"事后统计功效: {post_hoc_power:.4f}")

# === 7. 决策建议 ===
print("\n--- 步骤7: 决策建议 ---")
print("=" * 60)

is_significant = p_value < alpha
has_practical_significance = abs(h) >= 0.2

print(f"统计显著性: {'✓ 通过' if is_significant else '✗ 未通过'} (p={p_value:.4f})")
print(f"实际显著性: {'✓ 通过' if has_practical_significance else '✗ 效应量小'} (Cohen's h={h:.4f})")
print(f"统计功效: {'✓ 充足' if post_hoc_power >= 0.8 else '✗ 不足'} ({post_hoc_power:.4f})")

if is_significant and has_practical_significance:
    print(f"\n建议: 上线新版页面")
    print(f"  预期转化率提升: {diff:.2%} (CI: [{ci_lower:.2%}, {ci_upper:.2%}])")
    print(f"  日均10万UV，预计每日多转化{diff*100000:.0f}人")
elif is_significant and not has_practical_significance:
    print(f"\n建议: 统计显著但效应量小，需评估投入产出比")
elif not is_significant:
    print(f"\n建议: 未达到统计显著，不建议上线新版")
    if post_hoc_power < 0.8:
        additional_n = NormalIndPower().solve_power(
            effect_size=observed_effect, alpha=alpha, power=0.8, alternative='two-sided'
        )
        print(f"  功效不足，建议每组补充至{int(np.ceil(additional_n))}样本")
