# 数据来源: seaborn 内置 titanic 数据集
# 依赖库最低版本要求: pandas>=2.0, numpy>=1.24, matplotlib>=3.7, seaborn>=0.13, ydata-profiling>=4.0

import pandas as pd
import seaborn as sns
from pathlib import Path

# ============================================================
# 9. Pandas Profiling 自动化 EDA
# ydata-profiling (原 pandas-profiling) 可自动生成完整的 EDA 报告
# ============================================================

df = sns.load_dataset('titanic')
print("=" * 60)
print("泰坦尼克号数据集 - ydata-profiling 自动化 EDA")
print("=" * 60)

# --------------------------------------------------
# 9.1 生成 ProfileReport
# 一行代码即可生成完整的数据分析报告
# --------------------------------------------------
print("\n【9.1 生成 ProfileReport】")
print("正在生成分析报告,请稍候...")

from ydata_profiling import ProfileReport

profile = ProfileReport(df, title='泰坦尼克号数据集 EDA 报告', explorative=True)

# --------------------------------------------------
# 9.2 配置选项
# 通过参数控制报告的详细程度和内容
# --------------------------------------------------
print("\n【9.2 配置选项说明】")
print("  ProfileReport 主要参数:")
print("    title          - 报告标题")
print("    explorative    - 是否启用探索性分析模式(更详细)")
print("    minimal        - 是否生成精简版报告")
print("    sensitive      - 敏感数据模式,不显示具体值")
print("    sample         - 报告中显示的样本行数")
print("    correlations   - 相关性计算方法列表")
print("    missing_diagrams - 缺失值图表类型")

# 精简版报告示例
profile_minimal = ProfileReport(df, title='泰坦尼克号精简版报告', minimal=True)

# 自定义配置
profile_custom = ProfileReport(
    df,
    title='泰坦尼克号自定义报告',
    explorative=True,
    correlations={
        'auto': {'calculate': True},
        'pearson': {'calculate': True},
        'spearman': {'calculate': True},
    },
    missing_diagrams={
        'bar': True,
        'matrix': True,
        'heatmap': True,
    },
)

# --------------------------------------------------
# 9.3 保存报告为 HTML
# 将报告导出为独立的 HTML 文件,可在浏览器中查看
# --------------------------------------------------
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

html_path = output_dir / 'titanic_eda_report.html'
profile.to_file(html_path)
print(f"\n【9.3 报告已保存】")
print(f"  完整报告: {html_path}")

minimal_path = output_dir / 'titanic_eda_report_minimal.html'
profile_minimal.to_file(minimal_path)
print(f"  精简报告: {minimal_path}")

custom_path = output_dir / 'titanic_eda_report_custom.html'
profile_custom.to_file(custom_path)
print(f"  自定义报告: {custom_path}")

# --------------------------------------------------
# 9.4 报告主要章节说明
# ydata-profiling 生成的报告包含以下核心部分
# --------------------------------------------------
print("\n【9.4 报告主要章节】")
print("  1. 概览 (Overview)")
print("     - 数据集基本信息: 行数、列数、缺失值比例")
print("     - 变量类型分布")
print("     - 内存占用")
print()
print("  2. 变量详情 (Variables)")
print("     - 每个变量的统计描述")
print("     - 分布图(直方图/条形图)")
print("     - 极端值和缺失值信息")
print()
print("  3. 相关性 (Correlations)")
print("     - Pearson / Spearman / Kendall / Phik 相关系数矩阵")
print("     - 相关性热力图")
print()
print("  4. 缺失值 (Missing Values)")
print("     - 缺失值数量和比例")
print("     - 缺失值热力图和矩阵图")
print()
print("  5. 样本 (Sample)")
print("     - 数据集前/后若干行预览")
print()
print("  6. 重复行 (Duplicate Rows)")
print("     - 重复行数量和占比")
print()
print("  7. 交互分析 (Interactions)")
print("     - 变量间的交互散点图")

# --------------------------------------------------
# 补充: 以编程方式获取报告中的信息
# --------------------------------------------------
print("\n【补充: 获取报告描述信息】")
description = profile.get_description()
print(f"  数据集行数: {description.analysis.n_var}")
print(f"  数据集列数: {description.analysis.n}")
print(f"  缺失值比例: {description.analysis.p_missing:.2%}")
print(f"  重复行比例: {description.analysis.p_duplicates:.2%}")

print(f"\n报告已保存至: {output_dir}")
print("\n" + "=" * 60)
print("自动化 EDA 完成! ydata-profiling 大幅提升了 EDA 的效率。")
print("注意: 首次安装请运行 pip install ydata-profiling")
print("=" * 60)
