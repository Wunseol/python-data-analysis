# 依赖库最低版本要求: pandas>=2.0, nbformat>=5.0
# 数据来源: 本文件使用 pandas 构造的模拟数据集，无需外部数据文件
# 注意: 若需使用 papermill 参数化执行，需额外安装: pip install papermill

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path
from datetime import datetime

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("5. Jupyter Notebook自动化")
print("=" * 60)

# --------------------------------------------------
# 一、使用 nbformat 创建 Notebook
# --------------------------------------------------

nb = new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

print("[Notebook] 创建空白 Notebook")

# --------------------------------------------------
# 二、添加 Markdown 单元格
# --------------------------------------------------

nb.cells.append(new_markdown_cell("""# 员工数据分析报告 Notebook

> 自动生成日期: {date}

本 Notebook 由 Python 脚本自动生成，包含完整的数据分析流程。
""".format(date=datetime.now().strftime("%Y-%m-%d"))))

nb.cells.append(new_markdown_cell("## 一、数据准备\n\n导入所需库并生成模拟数据。"))

print("[Notebook] Markdown 单元格已添加")

# --------------------------------------------------
# 三、添加代码单元格 - 数据生成
# --------------------------------------------------

nb.cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
df = pd.DataFrame({
    '姓名': [f'员工{i:02d}' for i in range(1, 21)],
    '部门': np.random.choice(['技术部', '市场部', '财务部', '人事部'], 20),
    '月薪': np.random.randint(8000, 25000, 20),
    '绩效评分': np.round(np.random.uniform(60, 100, 20), 1),
    '入职日期': pd.date_range('2022-01-01', periods=20, freq='30D'),
})

print(f'数据维度: {df.shape}')
df.head()
"""))

print("[Notebook] 数据生成代码单元格已添加")

# --------------------------------------------------
# 四、添加代码单元格 - 数据分析
# --------------------------------------------------

nb.cells.append(new_markdown_cell("## 二、数据分析\n\n计算各部门汇总统计。"))

nb.cells.append(new_code_cell("""# 部门汇总统计
dept_summary = df.groupby('部门').agg(
    人数=('姓名', 'count'),
    平均月薪=('月薪', 'mean'),
    平均绩效=('绩效评分', 'mean'),
).round(2)

print('部门汇总:')
print(dept_summary)
print(f'\\n整体平均月薪: {df[\"月薪\"].mean():,.0f} 元')
print(f'整体平均绩效: {df[\"绩效评分\"].mean():.1f} 分')
"""))

nb.cells.append(new_markdown_cell("## 三、薪资分析\n\n分析薪资分布和部门差异。"))

nb.cells.append(new_code_cell("""# 薪资统计
print('薪资描述统计:')
print(df['月薪'].describe())

print(f'\\n薪资中位数: {df[\"月薪\"].median():,} 元')
print(f'薪资标准差: {df[\"月薪\"].std():,.0f} 元')
"""))

print("[Notebook] 数据分析代码单元格已添加")

# --------------------------------------------------
# 五、添加代码单元格 - 可视化
# --------------------------------------------------

nb.cells.append(new_markdown_cell("## 四、数据可视化"))

nb.cells.append(new_code_cell("""# 部门平均月薪柱状图
fig, ax = plt.subplots(figsize=(8, 5))
dept_avg = df.groupby('部门')['月薪'].mean().sort_values()
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
bars = ax.bar(dept_avg.index, dept_avg.values, color=colors[:len(dept_avg)])
for bar, val in zip(bars, dept_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:.0f}', ha='center', fontsize=10)
ax.set_ylabel('平均月薪 (元)')
ax.set_title('各部门平均月薪')
plt.tight_layout()
plt.show()
"""))

nb.cells.append(new_code_cell("""# 绩效评分分布直方图
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['绩效评分'], bins=8, color='#ED7D31', edgecolor='white')
ax.set_xlabel('绩效评分')
ax.set_ylabel('人数')
ax.set_title('绩效评分分布')
ax.axvline(df['绩效评分'].mean(), color='red', linestyle='--',
           label=f'均值: {df["绩效评分"].mean():.1f}')
ax.legend()
plt.tight_layout()
plt.show()
"""))

nb.cells.append(new_code_cell("""# 月薪与绩效散点图
fig, ax = plt.subplots(figsize=(8, 5))
departments = df['部门'].unique()
colors_map = {'技术部': '#4472C4', '市场部': '#ED7D31', '财务部': '#A5A5A5', '人事部': '#FFC000'}
for dept in departments:
    mask = df['部门'] == dept
    ax.scatter(df.loc[mask, '绩效评分'], df.loc[mask, '月薪'],
               label=dept, color=colors_map.get(dept, '#999'), s=60, alpha=0.7)
ax.set_xlabel('绩效评分')
ax.set_ylabel('月薪 (元)')
ax.set_title('月薪与绩效评分关系')
ax.legend()
plt.tight_layout()
plt.show()
"""))

print("[Notebook] 可视化代码单元格已添加")

# --------------------------------------------------
# 六、添加结论单元格
# --------------------------------------------------

nb.cells.append(new_markdown_cell("## 五、结论\n\n根据以上分析得出以下结论:"))

nb.cells.append(new_code_cell("""# 生成结论摘要
conclusions = []
conclusions.append(f'1. 共分析 {len(df)} 名员工，分布在 {df[\"部门\"].nunique()} 个部门')
conclusions.append(f'2. 平均月薪 {df[\"月薪\"].mean():,.0f} 元，薪资范围 {df[\"月薪\"].min()}-{df[\"月薪\"].max()} 元')
conclusions.append(f'3. 平均绩效评分 {df[\"绩效评分\"].mean():.1f} 分')

highest_dept = dept_summary['平均月薪'].idxmax()
conclusions.append(f'4. {highest_dept}平均薪资最高，为 {dept_summary.loc[highest_dept, \"平均月薪\"]:,.0f} 元')

for c in conclusions:
    print(c)
"""))

print("[Notebook] 结论单元格已添加")

# --------------------------------------------------
# 七、保存 Notebook
# --------------------------------------------------

nb_path = output_dir / "自动生成分析报告.ipynb"
nbformat.write(nb, str(nb_path))
print(f"\n[保存] Notebook已保存: {nb_path}")

print(f"[信息] 单元格总数: {len(nb.cells)}")
code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
md_cells = sum(1 for c in nb.cells if c.cell_type == "markdown")
print(f"[信息] 代码单元格: {code_cells}, Markdown单元格: {md_cells}")

# --------------------------------------------------
# 八、papermill 参数化执行示例
# --------------------------------------------------

param_nb = new_notebook()
param_nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

param_nb.cells.append(new_code_cell("""# Parameters (papermill 会替换此单元格的值)
seed = 42
num_employees = 20
"""))

param_nb.cells.append(new_code_cell("""import pandas as pd
import numpy as np

np.random.seed(seed)
df = pd.DataFrame({
    '姓名': [f'员工{i:02d}' for i in range(1, num_employees + 1)],
    '部门': np.random.choice(['技术部', '市场部', '财务部', '人事部'], num_employees),
    '月薪': np.random.randint(8000, 25000, num_employees),
})

print(f'员工数量: {len(df)}')
print(f'平均月薪: {df[\"月薪\"].mean():,.0f} 元')
df.head()
"""))

param_nb_path = output_dir / "参数化分析报告.ipynb"
nbformat.write(param_nb, str(param_nb_path))
print(f"\n[保存] 参数化Notebook已保存: {param_nb_path}")

print("\n--- papermill 使用说明 ---")
print("安装: pip install papermill")
print("执行: papermill 参数化分析报告.ipynb 输出报告.ipynb -p seed 100 -p num_employees 50")
print("papermill 会自动替换 Parameters 单元格中的变量值并执行整个 Notebook")

# --------------------------------------------------
# 九、验证 Notebook 结构
# --------------------------------------------------

print("\n--- Notebook 结构验证 ---")
loaded_nb = nbformat.read(str(nb_path), as_version=4)
for i, cell in enumerate(loaded_nb.cells):
    cell_type = cell.cell_type
    preview = cell.source[:60].replace("\n", " ")
    print(f"  [{i:2d}] {cell_type:8s} | {preview}...")

print("\n" + "=" * 60)
print("Jupyter Notebook自动化 - 完成")
print("=" * 60)
