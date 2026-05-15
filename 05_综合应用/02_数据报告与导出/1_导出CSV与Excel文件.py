# 依赖库最低版本要求: pandas>=2.0, openpyxl>=3.1, fpdf2>=2.7, nbformat>=5.0
# 数据来源: 本文件使用 pandas 构造的模拟数据集，无需外部数据文件

import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("1. 导出CSV与Excel文件")
print("=" * 60)

np.random.seed(42)
df = pd.DataFrame({
    "姓名": [f"员工{i:02d}" for i in range(1, 21)],
    "部门": np.random.choice(["技术部", "市场部", "财务部", "人事部"], 20),
    "月薪": np.random.randint(8000, 25000, 20),
    "绩效评分": np.round(np.random.uniform(60, 100, 20), 1),
    "入职日期": pd.date_range("2022-01-01", periods=20, freq="30D"),
})

print("\n原始数据预览:")
print(df.head())

# --------------------------------------------------
# 一、导出 CSV 文件
# --------------------------------------------------

csv_path = output_dir / "员工数据.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n[CSV] 已导出 (utf-8-sig编码): {csv_path}")

df.to_csv(output_dir / "员工数据_无BOM.csv", index=False, encoding="utf-8")
print(f"[CSV] 已导出 (utf-8编码，无BOM): {output_dir / '员工数据_无BOM.csv'}")

df.to_csv(output_dir / "员工数据_分号分隔.csv", index=False, encoding="utf-8-sig", sep=";")
print(f"[CSV] 已导出 (分号分隔): {output_dir / '员工数据_分号分隔.csv'}")

df.to_csv(output_dir / "员工数据_含索引.csv", encoding="utf-8-sig")
print(f"[CSV] 已导出 (含行索引): {output_dir / '员工数据_含索引.csv'}")

df.to_csv(
    output_dir / "员工数据_自定义缺失值.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="缺失",
)
print(f"[CSV] 已导出 (自定义缺失值标记): {output_dir / '员工数据_自定义缺失值.csv'}")

df_read = pd.read_csv(csv_path, encoding="utf-8-sig")
print("\n[CSV] 回读验证 - 前3行:")
print(df_read.head(3))

# --------------------------------------------------
# 二、导出 Excel 文件
# --------------------------------------------------

excel_path = output_dir / "员工数据.xlsx"
df.to_excel(excel_path, index=False, engine="openpyxl", sheet_name="员工信息")
print(f"\n[Excel] 已导出: {excel_path}")

df.to_excel(
    output_dir / "员工数据_多Sheet.xlsx",
    index=False,
    engine="openpyxl",
    sheet_name="全部员工",
)

dept_summary = df.groupby("部门").agg(
    人数=("姓名", "count"),
    平均月薪=("月薪", "mean"),
    平均绩效=("绩效评分", "mean"),
).round(2)

with pd.ExcelWriter(
    output_dir / "员工数据_多Sheet.xlsx", engine="openpyxl"
) as writer:
    df.to_excel(writer, sheet_name="原始数据", index=False)
    dept_summary.to_excel(writer, sheet_name="部门汇总")

print(f"[Excel] 多Sheet已导出: {output_dir / '员工数据_多Sheet.xlsx'}")

# --------------------------------------------------
# 三、处理中文文件名
# --------------------------------------------------

chinese_name_path = output_dir / "二〇二四年员工薪资表.xlsx"
df.to_excel(chinese_name_path, index=False, engine="openpyxl")
print(f"\n[Excel] 中文文件名导出成功: {chinese_name_path}")

chinese_csv_path = output_dir / "二〇二四年员工薪资表.csv"
df.to_csv(chinese_csv_path, index=False, encoding="utf-8-sig")
print(f"[CSV] 中文文件名导出成功: {chinese_csv_path}")

# --------------------------------------------------
# 四、使用 pathlib 管理输出路径
# --------------------------------------------------

project_root = Path(__file__).parent
reports_dir = project_root / "output" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
auto_named = reports_dir / f"员工数据_{timestamp}.csv"
df.to_csv(auto_named, index=False, encoding="utf-8-sig")
print(f"\n[pathlib] 自动命名导出: {auto_named}")

print("\n所有输出文件列表:")
for f in sorted(output_dir.rglob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(output_dir)} ({size_kb:.1f} KB)")

print("\n" + "=" * 60)
print("导出CSV与Excel文件 - 完成")
print("=" * 60)
