# 数据来源: 模拟数据
import numpy as np
from scipy import stats

np.random.seed(42)

# === 独立样本t检验 ===
print("=" * 50)
print("1. 独立样本t检验 — 两组用户停留时间对比")
print("=" * 50)

group_a = np.random.normal(120, 25, 200)  # 对照组均值120秒
group_b = np.random.normal(128, 25, 200)  # 实验组均值128秒

print(f"对照组: 均值={group_a.mean():.2f}, 标准差={group_a.std(ddof=1):.2f}")
print(f"实验组: 均值={group_b.mean():.2f}, 标准差={group_b.std(ddof=1):.2f}")

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t统计量: {t_stat:.4f}, p值: {p_val:.4f}")
print(f"结论: {'两组均值有显著差异' if p_val < 0.05 else '两组均值无显著差异'}")

# === 方差齐性检验 ===
print("\n" + "=" * 50)
print("2. 方差齐性检验 (Levene检验)")
print("=" * 50)

lev_stat, lev_p = stats.levene(group_a, group_b)
print(f"Levene统计量: {lev_stat:.4f}, p值: {lev_p:.4f}")
equal_var = lev_p >= 0.05
print(f"方差齐性: {'是' if equal_var else '否'} (p>0.05则齐性)")

# === Welch t检验 ===
print("\n" + "=" * 50)
print("3. Welch t检验 — 不假设方差齐性")
print("=" * 50)

group_c = np.random.normal(120, 20, 200)
group_d = np.random.normal(130, 40, 200)  # 方差不同

lev_stat2, lev_p2 = stats.levene(group_c, group_d)
print(f"Levene检验: p={lev_p2:.4f} → 方差{'齐性' if lev_p2 >= 0.05 else '不齐'}")

t_welch, p_welch = stats.ttest_ind(group_c, group_d, equal_var=False)
t_classic, p_classic = stats.ttest_ind(group_c, group_d, equal_var=True)
print(f"Welch t检验: t={t_welch:.4f}, p={p_welch:.4f}")
print(f"经典t检验: t={t_classic:.4f}, p={p_classic:.4f}")
print(f"方差不齐时Welch检验更可靠")

# === 配对t检验 ===
print("\n" + "=" * 50)
print("4. 配对t检验 — 同一用户前后对比")
print("=" * 50)

# 模拟用户改版前后的使用时长
n_users = 150
before = np.random.normal(100, 15, n_users)
after = before + np.random.normal(5, 8, n_users)  # 改版后平均增加5分钟

print(f"改版前均值: {before.mean():.2f}")
print(f"改版后均值: {after.mean():.2f}")
print(f"平均差异: {(after - before).mean():.2f}")

t_paired, p_paired = stats.ttest_rel(before, after)
print(f"配对t检验: t={t_paired:.4f}, p={p_paired:.4f}")

# 配对t检验等价于对差值做单样本t检验
diff = after - before
t_diff, p_diff = stats.ttest_1samp(diff, 0)
print(f"差值单样本t检验: t={t_diff:.4f}, p={p_diff:.4f} (结果一致)")

# === 各检验方法对比 ===
print("\n" + "=" * 50)
print("5. 检验方法选择指南")
print("=" * 50)
print("独立样本t检验: 两组独立样本均值对比，方差齐性")
print("Welch t检验:   两组独立样本均值对比，方差不齐")
print("配对t检验:     同一组样本前后/配对对比")
print("Levene检验:    先做方差齐性检验，再选择t检验类型")
