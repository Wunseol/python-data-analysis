# 依赖库最低版本要求: pandas>=2.0, fpdf2>=2.7, matplotlib>=3.7
# 数据来源: 本文件使用 pandas 构造的模拟数据集，无需外部数据文件

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from fpdf import FPDF

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("3. 生成PDF报告")
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
# 一、生成 matplotlib 图表
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

dept_avg = df.groupby("部门")["月薪"].mean().sort_values(ascending=True)
axes[0].barh(dept_avg.index, dept_avg.values, color="#4472C4")
axes[0].set_xlabel("平均月薪 (元)")
axes[0].set_title("各部门平均月薪")
for i, v in enumerate(dept_avg.values):
    axes[0].text(v + 200, i, f"{v:.0f}", va="center", fontsize=9)

axes[1].hist(df["绩效评分"], bins=8, color="#ED7D31", edgecolor="white")
axes[1].set_xlabel("绩效评分")
axes[1].set_ylabel("人数")
axes[1].set_title("绩效评分分布")
axes[1].axvline(df["绩效评分"].mean(), color="red", linestyle="--", label=f"均值: {df['绩效评分'].mean():.1f}")
axes[1].legend()

plt.tight_layout()
chart_path = output_dir / "部门薪资与绩效.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[图表] 已保存: {chart_path}")

# --------------------------------------------------
# 二、使用 fpdf2 创建 PDF 报告
# --------------------------------------------------

class ChinesePDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = Path(__file__).parent.parent.parent / "fonts"
        simhei = font_path / "SimHei.ttf"
        if simhei.exists():
            self.add_font("SimHei", "", str(simhei))
            self.add_font("SimHei", "B", str(simhei))
            self.cn_font = "SimHei"
        else:
            self.cn_font = "helvetica"

    def header(self):
        self.set_font(self.cn_font, "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "员工数据分析报告", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(68, 114, 196)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.cn_font, "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"第 {self.page_no()}/{{nb}} 页", align="C")

pdf = ChinesePDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# --------------------------------------------------
# 三、添加标题页
# --------------------------------------------------

pdf.add_page()
pdf.set_font(pdf.cn_font, "B", 24)
pdf.ln(40)
pdf.cell(0, 15, "员工数据分析报告", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font(pdf.cn_font, "", 14)
pdf.set_text_color(100, 100, 100)
pdf.ln(10)
pdf.cell(0, 10, f"报告日期: {pd.Timestamp.now().strftime('%Y年%m月%d日')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 10, "数据范围: 2022年度入职员工", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 10, f"样本数量: {len(df)} 人", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)

print("[PDF] 标题页已创建")

# --------------------------------------------------
# 四、添加摘要段落
# --------------------------------------------------

pdf.add_page()
pdf.set_font(pdf.cn_font, "B", 16)
pdf.cell(0, 12, "一、数据摘要", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

pdf.set_font(pdf.cn_font, "", 11)
summary_text = (
    f"本报告对 {len(df)} 名员工的数据进行了分析。"
    f"员工分布在 {df['部门'].nunique()} 个部门，"
    f"月薪范围为 {df['月薪'].min()}-{df['月薪'].max()} 元，"
    f"平均月薪为 {df['月薪'].mean():.0f} 元。"
    f"绩效评分均值为 {df['绩效评分'].mean():.1f} 分，"
    f"最高绩效评分为 {df['绩效评分'].max():.1f} 分。"
)
pdf.multi_cell(0, 8, summary_text)
pdf.ln(5)

print("[PDF] 摘要段落已添加")

# --------------------------------------------------
# 五、添加数据表格
# --------------------------------------------------

pdf.set_font(pdf.cn_font, "B", 16)
pdf.cell(0, 12, "二、部门汇总数据", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

dept_summary = df.groupby("部门").agg(
    人数=("姓名", "count"),
    平均月薪=("月薪", "mean"),
    平均绩效=("绩效评分", "mean"),
).round(2)

col_widths = [40, 30, 45, 45]
headers = ["部门", "人数", "平均月薪(元)", "平均绩效(分)"]

pdf.set_font(pdf.cn_font, "B", 10)
pdf.set_fill_color(68, 114, 196)
pdf.set_text_color(255, 255, 255)
for i, header in enumerate(headers):
    pdf.cell(col_widths[i], 10, header, border=1, fill=True, align="C")
pdf.ln()

pdf.set_font(pdf.cn_font, "", 10)
pdf.set_text_color(0, 0, 0)
fill = False
for dept, row in dept_summary.iterrows():
    if fill:
        pdf.set_fill_color(235, 241, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.cell(col_widths[0], 8, str(dept), border=1, fill=True, align="C")
    pdf.cell(col_widths[1], 8, str(int(row["人数"])), border=1, fill=True, align="C")
    pdf.cell(col_widths[2], 8, f"{row['平均月薪']:.2f}", border=1, fill=True, align="C")
    pdf.cell(col_widths[3], 8, f"{row['平均绩效']:.2f}", border=1, fill=True, align="C")
    pdf.ln()
    fill = not fill

print("[PDF] 数据表格已添加")

# --------------------------------------------------
# 六、嵌入图表
# --------------------------------------------------

pdf.ln(10)
pdf.set_font(pdf.cn_font, "B", 16)
pdf.cell(0, 12, "三、可视化分析", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

pdf.set_font(pdf.cn_font, "", 11)
pdf.multi_cell(0, 8, "下图展示了各部门平均薪资水平及绩效评分的分布情况:")
pdf.ln(3)

pdf.image(str(chart_path), x=10, w=190)

print("[PDF] 图表已嵌入")

# --------------------------------------------------
# 七、添加详细数据表
# --------------------------------------------------

pdf.add_page()
pdf.set_font(pdf.cn_font, "B", 16)
pdf.cell(0, 12, "四、员工明细数据", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

detail_cols = ["姓名", "部门", "月薪", "绩效评分"]
detail_widths = [30, 35, 40, 40]
pdf.set_font(pdf.cn_font, "B", 9)
pdf.set_fill_color(68, 114, 196)
pdf.set_text_color(255, 255, 255)
for i, header in enumerate(detail_cols):
    pdf.cell(detail_widths[i], 8, header, border=1, fill=True, align="C")
pdf.ln()

pdf.set_font(pdf.cn_font, "", 9)
pdf.set_text_color(0, 0, 0)
fill = False
for _, row in df.iterrows():
    if fill:
        pdf.set_fill_color(235, 241, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.cell(detail_widths[0], 7, str(row["姓名"]), border=1, fill=True, align="C")
    pdf.cell(detail_widths[1], 7, str(row["部门"]), border=1, fill=True, align="C")
    pdf.cell(detail_widths[2], 7, f"{row['月薪']:,} 元", border=1, fill=True, align="C")
    pdf.cell(detail_widths[3], 7, f"{row['绩效评分']:.1f}", border=1, fill=True, align="C")
    pdf.ln()
    fill = not fill

print("[PDF] 员工明细表已添加")

# --------------------------------------------------
# 八、输出 PDF
# --------------------------------------------------

pdf_path = output_dir / "员工数据分析报告.pdf"
pdf.output(str(pdf_path))
print(f"\n[保存] PDF报告已保存: {pdf_path}")
print(f"[保存] PDF文件大小: {pdf_path.stat().st_size / 1024:.1f} KB")

print("\n" + "=" * 60)
print("生成PDF报告 - 完成")
print("=" * 60)
