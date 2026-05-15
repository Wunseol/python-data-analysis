# 数据来源: 模拟数据
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

# === 多重比较问题 ===
print("=" * 50)
print("1. 多重比较问题 — 多次检验导致假阳性膨胀")
print("=" * 50)

n_tests = 20
alpha = 0.05
family_wise_error = 1 - (1 - alpha) ** n_tests
print(f"单次检验α: {alpha}")
print(f"检验次数: {n_tests}")
print(f"族错误率(FWER): {family_wise_error:.4f}")
print(f"含义: 做{n_tests}次检验，至少出现1次假阳性的概率为{family_wise_error:.2%}")

# 模拟: 20组A/A测试(真实无差异)，观察假阳性
n_sim = 5000
false_positives = 0
for _ in range(n_sim):
    p_values = []
    for _ in range(n_tests):
        a = np.random.normal(0, 1, 100)
        b = np.random.normal(0, 1, 100)
        _, p = stats.ttest_ind(a, b)
        p_values.append(p)
    if any(p < alpha for p in p_values):
        false_positives += 1

print(f"模拟FWER: {false_positives/n_sim:.4f} (理论值≈{family_wise_error:.4f})")

# === Bonferroni校正 ===
print("\n" + "=" * 50)
print("2. Bonferroni校正")
print("=" * 50)

# 模拟AB测试: 1组有真实差异，9组无差异
p_values = []
true_effects = [False] * 9 + [True]

for i in range(10):
    if true_effects[i]:
        a = np.random.normal(100, 15, 200)
        b = np.random.normal(103, 15, 200)
    else:
        a = np.random.normal(100, 15, 200)
        b = np.random.normal(100, 15, 200)
    _, p = stats.ttest_ind(a, b)
    p_values.append(p)

p_values = np.array(p_values)

print("未校正结果:")
for i, p in enumerate(p_values):
    sig = "显著*" if p < 0.05 else ""
    effect = "(有真实效应)" if true_effects[i] else "(无真实效应)"
    print(f"  检验{i+1}: p={p:.4f} {sig} {effect}")

# Bonferroni校正
reject_bonf, p_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print("\nBonferroni校正后:")
for i in range(len(p_values)):
    sig = "显著*" if reject_bonf[i] else ""
    effect = "(有真实效应)" if true_effects[i] else "(无真实效应)"
    print(f"  检验{i+1}: 校正p={p_bonf[i]:.4f} {sig} {effect}")

# === FDR Benjamini-Hochberg ===
print("\n" + "=" * 50)
print("3. FDR Benjamini-Hochberg校正")
print("=" * 50)

reject_bh, p_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print("BH校正后:")
for i in range(len(p_values)):
    sig = "显著*" if reject_bh[i] else ""
    effect = "(有真实效应)" if true_effects[i] else "(无真实效应)"
    print(f"  检验{i+1}: 校正p={p_bh[i]:.4f} {sig} {effect}")

# === 方法对比 ===
print("\n" + "=" * 50)
print("4. 校正方法对比")
print("=" * 50)

n_sig_uncorrected = (p_values < 0.05).sum()
n_sig_bonferroni = reject_bonf.sum()
n_sig_bh = reject_bh.sum()

print(f"未校正显著数: {n_sig_uncorrected}")
print(f"Bonferroni校正显著数: {n_sig_bonferroni}")
print(f"BH校正显著数: {n_sig_bh}")
print(f"\nBonferroni: 最保守，控制FWER，适合验证性研究")
print(f"BH: 较宽松，控制FDR，适合探索性研究")

# === A/A测试验证 ===
print("\n" + "=" * 50)
print("5. A/A测试验证 — 检验系统正确性")
print("=" * 50)

n_aa_tests = 100
n_aa_sig = 0
p_aa_list = []

for _ in range(n_aa_tests):
    a = np.random.normal(50, 10, 500)
    b = np.random.normal(50, 10, 500)
    _, p = stats.ttest_ind(a, b)
    p_aa_list.append(p)
    if p < 0.05:
        n_aa_sig += 1

print(f"A/A测试次数: {n_aa_tests}")
print(f"显著结果数: {n_aa_sig} (期望≈{n_aa_tests * 0.05:.1f})")
print(f"假阳性率: {n_aa_sig/n_aa_tests:.4f}")

# p值均匀性检验
ks_stat, ks_p = stats.kstest(p_aa_list, 'uniform')
print(f"p值均匀性KS检验: 统计量={ks_stat:.4f}, p值={ks_p:.4f}")
print(f"结论: {'p值分布均匀，实验系统正常' if ks_p > 0.05 else 'p值分布异常，需检查实验系统'}")
