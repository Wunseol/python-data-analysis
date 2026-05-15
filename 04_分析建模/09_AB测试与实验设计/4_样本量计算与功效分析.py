# 数据来源: 模拟数据
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

np.random.seed(42)

# === 比例检验样本量计算 ===
print("=" * 50)
print("1. 比例检验样本量计算")
print("=" * 50)

p1, p2 = 0.10, 0.13  # 基准转化率10%, 期望提升到13%
alpha = 0.05
power = 0.80

effect_size = proportion_effectsize(p2, p1)
print(f"基准转化率: {p1}")
print(f"期望转化率: {p2}")
print(f"效应量(Cohen's h): {effect_size:.4f}")

power_analysis = NormalIndPower()
n_per_group = power_analysis.solve_power(
    effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided'
)
print(f"每组所需样本量: {int(np.ceil(n_per_group))}")

# === 均值检验样本量计算 ===
print("\n" + "=" * 50)
print("2. 均值检验样本量计算")
print("=" * 50)

mu1, mu2, sigma = 120, 128, 25
d = (mu2 - mu1) / sigma  # Cohen's d
print(f"对照组均值: {mu1}, 实验组均值: {mu2}, 标准差: {sigma}")
print(f"效应量(Cohen's d): {d:.4f}")

tt_power = TTestIndPower()
n_mean = tt_power.solve_power(effect_size=d, alpha=alpha, power=power, alternative='two-sided')
print(f"每组所需样本量: {int(np.ceil(n_mean))}")

# === 最小可检测效应(MDE) ===
print("\n" + "=" * 50)
print("3. 最小可检测效应(MDE)")
print("=" * 50)

n_available = 500  # 每组可用样本量

mde = tt_power.solve_power(
    effect_size=None, alpha=alpha, power=power, nobs1=n_available, alternative='two-sided'
)
mde_absolute = mde * sigma
print(f"每组样本量: {n_available}")
print(f"最小可检测效应量(Cohen's d): {mde:.4f}")
print(f"最小可检测均值差异: {mde_absolute:.2f}秒")

# === 功效曲线 ===
print("\n" + "=" * 50)
print("4. 功效曲线 — 样本量与统计功效关系")
print("=" * 50)

sample_sizes = np.array([100, 200, 300, 400, 500, 600, 800, 1000])
print(f"{'样本量(每组)':>12} {'功效':>8}")
print("-" * 22)
for n in sample_sizes:
    pwr = tt_power.power(effect_size=d, nobs1=n, alpha=alpha, alternative='two-sided')
    print(f"{n:>12} {pwr:>8.4f}")

# === 不同参数组合的样本量 ===
print("\n" + "=" * 50)
print("5. 不同参数组合下的样本量需求")
print("=" * 50)

print(f"{'α':>6} {'功效':>6} {'效应量d':>8} {'样本量/组':>10}")
print("-" * 32)
for a in [0.01, 0.05, 0.10]:
    for pw in [0.80, 0.90, 0.95]:
        for es in [0.2, 0.5, 0.8]:
            n_req = tt_power.solve_power(effect_size=es, alpha=a, power=pw, alternative='two-sided')
            print(f"{a:>6.2f} {pw:>6.2f} {es:>8.2f} {int(np.ceil(n_req)):>10}")

# === 样本量公式验证 ===
print("\n" + "=" * 50)
print("6. 样本量公式手动验证")
print("=" * 50)

from scipy import stats

z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)
p_bar = (p1 + p2) / 2
n_formula = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) /
             (p2 - p1)) ** 2
print(f"公式计算每组样本量: {int(np.ceil(n_formula))}")
print(f"statsmodels计算: {int(np.ceil(n_per_group))}")
