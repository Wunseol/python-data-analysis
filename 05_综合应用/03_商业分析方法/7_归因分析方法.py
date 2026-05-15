# 数据来源: 模拟多渠道用户触点与转化数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

channels = ["搜索广告", "信息流", "社交媒体", "直接访问", "邮件营销", "KOL推荐"]
channel_weights = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]

n_users = 3000
n_conversions = 800

paths = []
for _ in range(n_conversions):
    path_len = np.random.randint(1, 6)
    path = list(np.random.choice(channels, size=path_len, p=channel_weights))
    revenue = round(np.random.exponential(200) + 50, 2)
    paths.append({"路径": " → ".join(path), "渠道列表": path, "收入": revenue})

df_paths = pd.DataFrame(paths)

print("=" * 60)
print("用户转化路径（前10条）")
print("=" * 60)
for i, row in df_paths.head(10).iterrows():
    print(f"  {row['路径']} | 收入: {row['收入']}")

def first_touch(paths_list, revenues):
    result = {}
    for path, rev in zip(paths_list, revenues):
        ch = path[0]
        result[ch] = result.get(ch, 0) + rev
    return result

def last_touch(paths_list, revenues):
    result = {}
    for path, rev in zip(paths_list, revenues):
        ch = path[-1]
        result[ch] = result.get(ch, 0) + rev
    return result

def linear(paths_list, revenues):
    result = {}
    for path, rev in zip(paths_list, revenues):
        n = len(path)
        for ch in path:
            result[ch] = result.get(ch, 0) + rev / n
    return result

def time_decay(paths_list, revenues, decay_rate=0.5):
    result = {}
    for path, rev in zip(paths_list, revenues):
        n = len(path)
        weights = [decay_rate ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        for ch, w in zip(path, weights):
            result[ch] = result.get(ch, 0) + rev * w / total_w
    return result

paths_list = df_paths["渠道列表"].tolist()
revenues = df_paths["收入"].tolist()

first = first_touch(paths_list, revenues)
last = last_touch(paths_list, revenues)
lin = linear(paths_list, revenues)
decay = time_decay(paths_list, revenues)

total_rev = sum(revenues)

attribution = pd.DataFrame({
    "渠道": channels,
    "首次归因": [first.get(ch, 0) for ch in channels],
    "末次归因": [last.get(ch, 0) for ch in channels],
    "线性归因": [lin.get(ch, 0) for ch in channels],
    "时间衰减归因": [decay.get(ch, 0) for ch in channels],
})

for col in ["首次归因", "末次归因", "线性归因", "时间衰减归因"]:
    attribution[f"{col}_占比"] = (attribution[col] / total_rev).round(4)

print("\n" + "=" * 60)
print("归因分析结果")
print("=" * 60)
print(attribution[["渠道", "首次归因", "末次归因", "线性归因", "时间衰减归因"]].round(2).to_string(index=False))

print("\n渠道贡献度占比:")
print(attribution[["渠道", "首次归因_占比", "末次归因_占比", "线性归因_占比", "时间衰减归因_占比"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

x = np.arange(len(channels))
width = 0.2
methods = ["首次归因", "末次归因", "线性归因", "时间衰减归因"]
colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]

for i, method in enumerate(methods):
    axes[0].bar(x + i * width, attribution[method], width, label=method, color=colors[i])

axes[0].set_xticks(x + 1.5 * width)
axes[0].set_xticklabels(channels, rotation=30)
axes[0].set_ylabel("归因收入")
axes[0].set_title("各归因模型渠道贡献对比")
axes[0].legend()

pct_cols = [f"{m}_占比" for m in methods]
bottom = np.zeros(len(channels))
for i, (col, color) in enumerate(zip(pct_cols, colors)):
    axes[1].bar(channels, attribution[col], bottom=bottom, label=methods[i], color=color)
    bottom += attribution[col].values

axes[1].set_ylabel("贡献占比")
axes[1].set_title("渠道贡献度堆叠对比")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "归因分析.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存至: {OUTPUT_DIR / '归因分析.png'}")
