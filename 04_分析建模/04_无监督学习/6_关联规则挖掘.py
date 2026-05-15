# 数据来源: 自建交易数据 (模拟超市购物篮)

"""
关联规则挖掘
============
本案例演示:
1. Apriori 算法概念
2. mlxtend 库 (如可用) 或手动实现
3. 支持度 (support)、置信度 (confidence)、提升度 (lift)
4. 交易数据格式
5. 频繁项集挖掘
6. 规则生成与解读
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent

# ============================================================
# 一、构建交易数据
# ============================================================

transactions = [
    ['面包', '牛奶', '啤酒'],
    ['面包', '尿布', '啤酒', '鸡蛋'],
    ['牛奶', '尿布', '啤酒', '可乐'],
    ['面包', '牛奶', '尿布', '啤酒'],
    ['面包', '牛奶', '尿布', '可乐'],
    ['面包', '牛奶', '尿布'],
    ['面包', '啤酒', '可乐'],
    ['牛奶', '尿布', '可乐'],
    ['面包', '牛奶', '尿布', '啤酒', '鸡蛋'],
    ['尿布', '啤酒', '可乐'],
    ['面包', '牛奶', '鸡蛋'],
    ['面包', '牛奶', '尿布', '可乐'],
    ['面包', '啤酒', '鸡蛋'],
    ['牛奶', '尿布', '啤酒', '鸡蛋'],
    ['面包', '牛奶', '尿布', '啤酒', '可乐'],
]

print("=" * 60)
print("【交易数据】")
print(f"交易总数: {len(transactions)}")
for i, t in enumerate(transactions):
    print(f"  T{i + 1}: {t}")

all_items = sorted(set(item for t in transactions for item in t))
print(f"\n所有商品: {all_items}")
print(f"商品总数: {len(all_items)}")

# ============================================================
# 二、交易数据格式转换 (One-Hot 编码)
# ============================================================

def transactions_to_onehot(trans_list, items_list):
    n_trans = len(trans_list)
    n_items = len(items_list)
    matrix = np.zeros((n_trans, n_items), dtype=int)
    for i, trans in enumerate(trans_list):
        for item in trans:
            if item in items_list:
                j = items_list.index(item)
                matrix[i, j] = 1
    return matrix

onehot_matrix = transactions_to_onehot(transactions, all_items)

print("\n" + "=" * 60)
print("【One-Hot 编码矩阵】")
header = "交易ID  " + "  ".join(f"{item:<4}" for item in all_items)
print(header)
print("-" * len(header))
for i, row in enumerate(onehot_matrix):
    row_str = "  ".join(f"  {val} " for val in row)
    print(f"T{i + 1:>5}  {row_str}")

# ============================================================
# 三、手动实现 Apriori 算法
# ============================================================

def get_support(itemset, trans_list):
    count = sum(1 for t in trans_list if set(itemset).issubset(set(t)))
    return count / len(trans_list)

def find_frequent_1_itemsets(trans_list, items_list, min_support):
    frequent = {}
    for item in items_list:
        sup = get_support([item], trans_list)
        if sup >= min_support:
            frequent[frozenset([item])] = sup
    return frequent

def apriori_gen(prev_itemsets, k):
    candidates = set()
    items_list = sorted(set(item for s in prev_itemsets for item in s))
    for combo in combinations(items_list, k):
        candidates.add(frozenset(combo))
    return candidates

def apriori(trans_list, items_list, min_support):
    all_frequent = {}
    freq_k = find_frequent_1_itemsets(trans_list, items_list, min_support)
    all_frequent.update(freq_k)

    k = 2
    while freq_k:
        candidates = apriori_gen(freq_k.keys(), k)
        freq_k = {}
        for candidate in candidates:
            subsets = [frozenset(s) for s in combinations(candidate, k - 1)]
            if all(s in all_frequent for s in subsets):
                sup = get_support(list(candidate), trans_list)
                if sup >= min_support:
                    freq_k[candidate] = sup
        all_frequent.update(freq_k)
        k += 1

    return all_frequent

min_support = 0.3
frequent_itemsets = apriori(transactions, all_items, min_support)

print("\n" + "=" * 60)
print(f"【频繁项集 (最小支持度={min_support})】")
print(f"{'项集':<30} {'支持度':>10} {'频次':>8}")
print("-" * 50)

for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (len(x[0]), x[0])):
    freq = int(support * len(transactions))
    items_str = "{" + ", ".join(sorted(itemset)) + "}"
    print(f"{items_str:<30} {support:>10.4f} {freq:>8}")

# ============================================================
# 四、关联规则生成
# ============================================================

def generate_rules(frequent_itemsets, trans_list, min_confidence=0.5):
    rules = []
    for itemset, support_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                support_x = get_support(list(antecedent), trans_list)
                support_y = get_support(list(consequent), trans_list)
                confidence = support_xy / support_x if support_x > 0 else 0
                lift = confidence / support_y if support_y > 0 else 0

                if confidence >= min_confidence:
                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': support_xy,
                        'confidence': confidence,
                        'lift': lift,
                    })
    return rules

min_confidence = 0.5
rules = generate_rules(frequent_itemsets, transactions, min_confidence)

print("\n" + "=" * 60)
print(f"【关联规则 (最小置信度={min_confidence})】")
print(f"{'前件':<20} {'后件':<15} {'支持度':>8} {'置信度':>8} {'提升度':>8}")
print("-" * 65)

rules_sorted = sorted(rules, key=lambda r: r['lift'], reverse=True)
for rule in rules_sorted:
    ant_str = "{" + ", ".join(sorted(rule['antecedent'])) + "}"
    con_str = "{" + ", ".join(sorted(rule['consequent'])) + "}"
    print(f"{ant_str:<20} {con_str:<15} {rule['support']:>8.4f} {rule['confidence']:>8.4f} {rule['lift']:>8.4f}")

# ============================================================
# 五、指标解读
# ============================================================

print("\n" + "=" * 60)
print("【关联规则指标解读】")
print("支持度 (Support): 项集在所有交易中出现的频率")
print("  - 支持度(X→Y) = P(X∪Y) = 包含X和Y的交易数 / 总交易数")
print("  - 衡量规则的普遍程度")
print()
print("置信度 (Confidence): 包含X的交易中也包含Y的概率")
print("  - 置信度(X→Y) = P(Y|X) = 支持度(X∪Y) / 支持度(X)")
print("  - 衡量规则的可靠程度")
print()
print("提升度 (Lift): X出现时Y出现的概率是Y单独出现概率的多少倍")
print("  - 提升度(X→Y) = 置信度(X→Y) / 支持度(Y)")
print("  - 提升度>1: 正相关 (X促进Y)")
print("  - 提升度=1: 无关联")
print("  - 提升度<1: 负相关 (X抑制Y)")

print("\n【高提升度规则解读】")
for rule in rules_sorted[:5]:
    ant_str = ", ".join(sorted(rule['antecedent']))
    con_str = ", ".join(sorted(rule['consequent']))
    if rule['lift'] > 1:
        print(f"  购买[{ant_str}]的客户购买[{con_str}]的概率是平均的{rule['lift']:.2f}倍 (正相关)")
    elif rule['lift'] < 1:
        print(f"  购买[{ant_str}]的客户购买[{con_str}]的概率是平均的{rule['lift']:.2f}倍 (负相关)")
    else:
        print(f"  购买[{ant_str}]与购买[{con_str}]无关联")

# ============================================================
# 六、频繁项集可视化
# ============================================================

itemset_names = []
itemset_supports = []
itemset_sizes = []

for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (len(x[0]), -x[1])):
    items_str = "{" + ", ".join(sorted(itemset)) + "}"
    itemset_names.append(items_str)
    itemset_supports.append(support)
    itemset_sizes.append(len(itemset))

fig, ax = plt.subplots(figsize=(14, 7))

colors = plt.cm.Set2(np.linspace(0, 1, max(itemset_sizes)))
bar_colors = [colors[s - 1] for s in itemset_sizes]

bars = ax.barh(range(len(itemset_names)), itemset_supports, color=bar_colors, alpha=0.8)

ax.set_yticks(range(len(itemset_names)))
ax.set_yticklabels(itemset_names, fontsize=9)
ax.set_xlabel('支持度', fontsize=12)
ax.set_title(f'频繁项集支持度 (最小支持度={min_support})', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

for bar, sup in zip(bars, itemset_supports):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{sup:.3f}', va='center', fontsize=9)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], alpha=0.8, label=f'{i + 1}-项集')
                   for i in range(max(itemset_sizes))]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '关联规则_频繁项集.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 七、关联规则可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

supports = [r['support'] for r in rules_sorted]
confidences = [r['confidence'] for r in rules_sorted]
lifts = [r['lift'] for r in rules_sorted]

sc = axes[0].scatter(supports, confidences, c=lifts, cmap='RdYlGn',
                      s=80, alpha=0.8, edgecolors='gray', linewidths=0.5)
plt.colorbar(sc, ax=axes[0], label='提升度')
axes[0].set_xlabel('支持度', fontsize=12)
axes[0].set_ylabel('置信度', fontsize=12)
axes[0].set_title('支持度 vs 置信度 (颜色=提升度)', fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(supports, lifts, c=confidences, cmap='RdYlGn',
                s=80, alpha=0.8, edgecolors='gray', linewidths=0.5)
plt.colorbar(axes[1].collections[0], ax=axes[1], label='置信度')
axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='提升度=1')
axes[1].set_xlabel('支持度', fontsize=12)
axes[1].set_ylabel('提升度', fontsize=12)
axes[1].set_title('支持度 vs 提升度 (颜色=置信度)', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].scatter(confidences, lifts, c=supports, cmap='RdYlGn',
                s=80, alpha=0.8, edgecolors='gray', linewidths=0.5)
plt.colorbar(axes[2].collections[0], ax=axes[2], label='支持度')
axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='提升度=1')
axes[2].set_xlabel('置信度', fontsize=12)
axes[2].set_ylabel('提升度', fontsize=12)
axes[2].set_title('置信度 vs 提升度 (颜色=支持度)', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle('关联规则: 支持度/置信度/提升度 散点图', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '关联规则_散点图.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 八、不同最小支持度的影响
# ============================================================

support_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
n_frequent = []
n_rules_list = []

for min_sup in support_thresholds:
    freq = apriori(transactions, all_items, min_sup)
    n_frequent.append(len(freq))
    r = generate_rules(freq, transactions, min_confidence=0.5)
    n_rules_list.append(len(r))

print("\n" + "=" * 60)
print("【不同最小支持度的影响】")
print(f"{'最小支持度':<14} {'频繁项集数':>12} {'关联规则数':>12}")
print("-" * 40)
for ms, nf, nr in zip(support_thresholds, n_frequent, n_rules_list):
    print(f"{ms:<14.1f} {nf:>12} {nr:>12}")

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'steelblue'
ax1.bar([x - 0.02 for x in support_thresholds], n_frequent, width=0.04,
        color=color1, alpha=0.8, label='频繁项集数')
ax1.set_xlabel('最小支持度', fontsize=12)
ax1.set_ylabel('频繁项集数', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'darkorange'
ax2.bar([x + 0.02 for x in support_thresholds], n_rules_list, width=0.04,
        color=color2, alpha=0.8, label='关联规则数')
ax2.set_ylabel('关联规则数', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

ax1.set_xticks(support_thresholds)
ax1.set_title('最小支持度对频繁项集和关联规则数量的影响', fontsize=14)
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '关联规则_支持度影响.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 九、商品共现热力图
# ============================================================

n_items = len(all_items)
co_occurrence = np.zeros((n_items, n_items), dtype=int)

for trans in transactions:
    for i, item1 in enumerate(all_items):
        for j, item2 in enumerate(all_items):
            if item1 in trans and item2 in trans:
                co_occurrence[i, j] += 1

co_ratio = co_occurrence / len(transactions)

print("\n" + "=" * 60)
print("【商品共现矩阵 (共现比例)】")
header = "        " + "  ".join(f"{item:<6}" for item in all_items)
print(header)
for i, item in enumerate(all_items):
    row_str = "  ".join(f"{co_ratio[i, j]:>6.3f}" for j in range(n_items))
    print(f"{item:<8}{row_str}")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(co_ratio, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(n_items))
ax.set_yticks(range(n_items))
ax.set_xticklabels(all_items, fontsize=11)
ax.set_yticklabels(all_items, fontsize=11)

for i in range(n_items):
    for j in range(n_items):
        ax.text(j, i, f'{co_ratio[i, j]:.2f}', ha='center', va='center',
                fontsize=9, color='black' if co_ratio[i, j] < 0.5 else 'white')

plt.colorbar(im, ax=ax, label='共现比例')
ax.set_title('商品共现热力图', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '关联规则_共现热力图.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 十、使用 mlxtend 库 (如可用)
# ============================================================

try:
    from mlxtend.frequent_patterns import apriori as mlxtend_apriori
    from mlxtend.frequent_patterns import association_rules as mlxtend_rules
    import pandas as pd

    df = pd.DataFrame(onehot_matrix, columns=all_items)

    freq_itemsets = mlxtend_apriori(df, min_support=0.3, use_colnames=True)
    freq_itemsets['length'] = freq_itemsets['itemsets'].apply(len)

    print("\n" + "=" * 60)
    print("【mlxtend 频繁项集 (min_support=0.3)】")
    print(freq_itemsets.to_string(index=False))

    assoc_rules = mlxtend_rules(freq_itemsets, metric='confidence', min_threshold=0.5)

    print("\n" + "=" * 60)
    print("【mlxtend 关联规则 (min_confidence=0.5)】")
    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(assoc_rules[cols].to_string(index=False))

except ImportError:
    print("\n" + "=" * 60)
    print("【mlxtend 库未安装】")
    print("mlxtend 库未安装, 如需使用请执行: pip install mlxtend")
    print("mlxtend 提供了高效的 Apriori 和 FP-Growth 实现")
    print("本案例已包含手动实现, 功能完整")

print("\n" + "=" * 60)
print("关联规则挖掘演示完成!")
