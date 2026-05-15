# 数据来源: 模拟数据
# 依赖库最低版本要求: scipy>=1.10, statsmodels>=0.14, numpy>=1.24
import numpy as np
from scipy import stats

np.random.seed(42)

# === AB测试基本流程 ===
print("=" * 50)
print("1. AB测试基本流程")
print("=" * 50)
print("步骤: 确定指标 → 设定假设 → 收集数据 → 统计检验 → 决策")

# === 零假设与备择假设 ===
print("\n" + "=" * 50)
print("2. 零假设(H0)与备择假设(H1)")
print("=" * 50)

# 模拟AB测试: 旧版转化率 vs 新版转化率
n_a, n_b = 1000, 1000
p_a, p_b = 0.10, 0.13  # 旧版10%, 新版13%

conversions_a = np.random.binomial(1, p_a, n_a)
conversions_b = np.random.binomial(1, p_b, n_b)

rate_a = conversions_a.mean()
rate_b = conversions_b.mean()

print(f"零假设H0: p_A = p_B (两组转化率无差异)")
print(f"备择假设H1: p_A ≠ p_B (两组转化率有差异)")
print(f"对照组转化率: {rate_a:.4f}")
print(f"实验组转化率: {rate_b:.4f}")
print(f"观测差异: {rate_b - rate_a:.4f}")

# === p值含义 ===
print("\n" + "=" * 50)
print("3. p值含义与计算")
print("=" * 50)

# 两比例z检验
p_pool = (conversions_a.sum() + conversions_b.sum()) / (n_a + n_b)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
z_stat = (rate_b - rate_a) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"合并比例: {p_pool:.4f}")
print(f"标准误: {se:.4f}")
print(f"z统计量: {z_stat:.4f}")
print(f"p值: {p_value:.4f}")
print(f"p值含义: 在H0为真时，观测到当前或更极端结果的概率")

# 使用scipy验证
count = np.array([conversions_a.sum(), conversions_b.sum()])
nobs = np.array([n_a, n_b])
z_check, p_check = stats.proportions_ztest(count, nobs, alternative='two-sided')
print(f"scipy验证 z={z_check:.4f}, p={p_check:.4f}")

# === 第一类错误与第二类错误 ===
print("\n" + "=" * 50)
print("4. 第一类错误(α)与第二类错误(β)")
print("=" * 50)

alpha = 0.05
print(f"第一类错误(α): H0为真时错误拒绝H0的概率 (假阳性)")
print(f"  显著性水平α = {alpha}")
print(f"第二类错误(β): H0为假时未能拒绝H0的概率 (假阴性)")
print(f"  统计功效 = 1 - β = 正确拒绝错误H0的概率")

# 模拟第一类错误率: 当两组真实无差异时，拒绝H0的比例
n_sim = 10000
type1_errors = 0
for _ in range(n_sim):
    a = np.random.binomial(1, 0.10, 500)
    b = np.random.binomial(1, 0.10, 500)
    _, p = stats.proportions_ztest([a.sum(), b.sum()], [500, 500])
    if p < alpha:
        type1_errors += 1

print(f"\n模拟验证第一类错误率: {type1_errors/n_sim:.4f} (理论值≈{alpha})")

# 模拟第二类错误率: 当两组真实有差异时，未能拒绝H0的比例
type2_errors = 0
for _ in range(n_sim):
    a = np.random.binomial(1, 0.10, 500)
    b = np.random.binomial(1, 0.13, 500)
    _, p = stats.proportions_ztest([a.sum(), b.sum()], [500, 500])
    if p >= alpha:
        type2_errors += 1

print(f"模拟验证第二类错误率: {type2_errors/n_sim:.4f}")
print(f"模拟统计功效: {1 - type2_errors/n_sim:.4f}")

# === 显著性水平α与决策 ===
print("\n" + "=" * 50)
print("5. 显著性水平α与统计决策")
print("=" * 50)

for alpha in [0.01, 0.05, 0.10]:
    decision = "拒绝H0" if p_value < alpha else "不拒绝H0"
    print(f"α={alpha:.2f}: p值={p_value:.4f} → {decision}")

print(f"\n结论: 在α=0.05水平下，{'新版转化率显著高于旧版' if p_value < 0.05 else '未发现显著差异'}")
