# 依赖库最低版本要求: pandas>=2.0, matplotlib>=3.7
# 数据来源: 本文件使用 pandas 构造的模拟数据集，无需外部数据文件

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("4. Markdown报告自动生成")
print("=" * 60)

np.random.seed(42)
df = pd.DataFrame({
    "姓名": [f"员工{i:02d}" for i in range(1, 21)],
    "部门": np.random.choice(["技术部", "市场部", "财务部", "人事部"], 20),
    "月薪": np.random.randint(8000, 25000, 20),
    "绩效评分": np.round(np.random.uniform(60, 100, 20), 1),
    "入职日期": pd.date_range("2022-01-01", periods=20, freq="30D"),
})

# --------------------------------------------------
# 一、生成图表并保存为图片
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
dept_avg = df.groupby("部门")["月薪"].mean().sort_values()
colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000"]
bars = ax.bar(dept_avg.index, dept_avg.values, color=colors[:len(dept_avg)])
for bar, val in zip(bars, dept_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:.0f}", ha="center", fontsize=10)
ax.set_ylabel("平均月薪 (元)")
ax.set_title("各部门平均月薪对比")
plt.tight_layout()
chart_path = output_dir / "markdown_chart.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[图表] 已保存: {chart_path}")

# --------------------------------------------------
# 二、构建 Markdown 报告内容
# --------------------------------------------------

report_date = datetime.now().strftime("%Y年%m月%d日")

title = f"# 员工数据分析报告\n\n> 生成日期: {report_date}\n"

summary_section = f"""
## 一、数据摘要

本报告对 **{len(df)}** 名员工数据进行了分析，涵盖 **{df['部门'].nunique()}** 个部门。

| 指标 | 数值 |
|------|------|
| 员工总数 | {len(df)} 人 |
| 平均月薪 | {df['月薪'].mean():,.0f} 元 |
| 最低月薪 | {df['月薪'].min():,} 元 |
| 最高月薪 | {df['月薪'].max():,} 元 |
| 平均绩效评分 | {df['绩效评分'].mean():.1f} 分 |
| 最高绩效评分 | {df['绩效评分'].max():.1f} 分 |
"""

dept_summary = df.groupby("部门").agg(
    人数=("姓名", "count"),
    平均月薪=("月薪", "mean"),
    平均绩效=("绩效评分", "mean"),
).round(2)

dept_section = """
## 二、部门汇总

| 部门 | 人数 | 平均月薪(元) | 平均绩效(分) |
|------|------|-------------|-------------|
"""
for dept, row in dept_summary.iterrows():
    dept_section += f"| {dept} | {int(row['人数'])} | {row['平均月薪']:,.2f} | {row['平均绩效']:.2f} |\n"

chart_section = f"""
## 三、可视化分析

![各部门平均月薪对比]({chart_path.name})

*图: 各部门平均月薪对比图*
"""

top5 = df.nlargest(5, "月薪")[["姓名", "部门", "月薪", "绩效评分"]]
top5_section = """
## 四、薪资TOP5

| 排名 | 姓名 | 部门 | 月薪(元) | 绩效评分 |
|------|------|------|---------|---------|
"""
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    top5_section += f"| {rank} | {row['姓名']} | {row['部门']} | {row['月薪']:,} | {row['绩效评分']:.1f} |\n"

bottom5 = df.nsmallest(5, "绩效评分")[["姓名", "部门", "月薪", "绩效评分"]]
attention_section = """
## 五、需关注员工 (绩效评分最低5人)

| 姓名 | 部门 | 月薪(元) | 绩效评分 |
|------|------|---------|---------|
"""
for _, row in bottom5.iterrows():
    attention_section += f"| {row['姓名']} | {row['部门']} | {row['月薪']:,} | {row['绩效评分']:.1f} |\n"

conclusion_section = f"""
## 六、结论与建议

1. **薪资分布**: 技术部平均薪资最高，建议关注各部门薪资公平性
2. **绩效分析**: 整体绩效均值为 {df['绩效评分'].mean():.1f} 分，部分员工绩效偏低需关注
3. **数据质量**: 共 {len(df)} 条记录，数据完整无缺失值

---

*本报告由 Python 自动生成*
"""

# --------------------------------------------------
# 三、组合完整报告并保存
# --------------------------------------------------

full_report = title + summary_section + dept_section + chart_section + top5_section + attention_section + conclusion_section

md_path = output_dir / "员工数据分析报告.md"
md_path.write_text(full_report, encoding="utf-8")
print(f"\n[保存] Markdown报告已保存: {md_path}")

# --------------------------------------------------
# 四、使用 string.Template 模板方式
# --------------------------------------------------

from string import Template

template_str = Template("""# $title

> 生成日期: $date
> 数据来源: $source

## 概述

$total_summary

## 关键指标

$key_metrics

## 部门数据

$dept_data

---

*本报告由 Python 自动生成*
""")

total_summary = f"本次分析共涵盖 {len(df)} 名员工，分布在 {df['部门'].nunique()} 个部门。"

key_metrics = f"""
- 平均月薪: **{df['月薪'].mean():,.0f}** 元
- 薪资中位数: **{df['月薪'].median():,.0f}** 元
- 平均绩效: **{df['绩效评分'].mean():.1f}** 分
"""

dept_lines = []
for dept, row in dept_summary.iterrows():
    dept_lines.append(f"- **{dept}**: {int(row['人数'])}人, 平均月薪 {row['平均月薪']:,.0f} 元")
dept_data = "\n".join(dept_lines)

template_report = template_str.substitute(
    title="员工数据快速报告",
    date=report_date,
    source="内部HR系统",
    total_summary=total_summary,
    key_metrics=key_metrics,
    dept_data=dept_data,
)

template_path = output_dir / "员工数据快速报告_模板.md"
template_path.write_text(template_report, encoding="utf-8")
print(f"[保存] 模板式Markdown报告已保存: {template_path}")

# --------------------------------------------------
# 五、验证输出
# --------------------------------------------------

print("\n--- Markdown报告内容预览 (前30行) ---")
lines = full_report.split("\n")
for i, line in enumerate(lines[:30], 1):
    print(f"{i:3d} | {line}")
if len(lines) > 30:
    print(f"... (共 {len(lines)} 行)")

print("\n--- 模板式报告内容 ---")
print(template_report)

print("\n" + "=" * 60)
print("Markdown报告自动生成 - 完成")
print("=" * 60)
