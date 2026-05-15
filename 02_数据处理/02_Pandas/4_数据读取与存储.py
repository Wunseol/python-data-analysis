# 数据来源: 脚本内自建示例数据
# 本脚本演示 Pandas 数据读取与存储方法

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

df_employees = pd.DataFrame({
    '工号': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008'],
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十'],
    '部门': ['技术部', '市场部', '技术部', '财务部', '市场部', '技术部', '财务部', '市场部'],
    '薪资': [15000, 22000, 18000, 25000, 12000, 20000, 23000, 16000],
    '入职年份': [2019, 2017, 2020, 2016, 2022, 2018, 2015, 2021]
})

df_scores = pd.DataFrame({
    '学号': [1001, 1002, 1003, 1004, 1005],
    '语文': [85, 92, 78, 90, 88],
    '数学': [90, 76, 95, 82, 91],
    '英语': [78, 88, 82, 85, 79]
})

print("员工数据:")
print(df_employees)
print("\n成绩数据:")
print(df_scores)

print("\n" + "=" * 60)
print("二、CSV 文件写入与读取")
print("=" * 60)

csv_path = OUTPUT_DIR / 'employees.csv'
df_employees.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\n已保存 CSV 文件: {csv_path}")

# 基本读取
df_read = pd.read_csv(csv_path)
print(f"\n基本读取:\n{df_read}")

# 指定编码
df_read_enc = pd.read_csv(csv_path, encoding='utf-8-sig')
print(f"\n指定编码读取:\n{df_read_enc.head(3)}")

# 指定分隔符 - 先创建用分号分隔的文件
csv_sep_path = OUTPUT_DIR / 'employees_sep.csv'
df_employees.to_csv(csv_sep_path, index=False, sep=';', encoding='utf-8-sig')
df_read_sep = pd.read_csv(csv_sep_path, sep=';', encoding='utf-8-sig')
print(f"\n指定分隔符读取 (sep=';'):\n{df_read_sep.head(3)}")

# header 参数
csv_no_header_path = OUTPUT_DIR / 'employees_no_header.csv'
df_employees.to_csv(csv_no_header_path, index=False, header=False, encoding='utf-8-sig')
df_read_no_header = pd.read_csv(csv_no_header_path, header=None, encoding='utf-8-sig')
print(f"\n无表头文件读取 (header=None):\n{df_read_no_header.head(3)}")

# 指定列名
df_read_names = pd.read_csv(csv_no_header_path, header=None,
                            names=['工号', '姓名', '部门', '薪资', '入职年份'],
                            encoding='utf-8-sig')
print(f"\n指定列名读取:\n{df_read_names.head(3)}")

# index_col 参数
df_read_idx = pd.read_csv(csv_path, index_col='工号', encoding='utf-8-sig')
print(f"\n指定索引列 (index_col='工号'):\n{df_read_idx.head(3)}")

# usecols 参数 - 只读取部分列
df_read_cols = pd.read_csv(csv_path, usecols=['姓名', '薪资'], encoding='utf-8-sig')
print(f"\n只读取部分列 (usecols):\n{df_read_cols.head(3)}")

# nrows 参数 - 只读取前N行
df_read_nrows = pd.read_csv(csv_path, nrows=3, encoding='utf-8-sig')
print(f"\n只读取前3行 (nrows=3):\n{df_read_nrows}")

print("\n" + "=" * 60)
print("三、Excel 文件写入与读取")
print("=" * 60)

excel_path = OUTPUT_DIR / 'data.xlsx'

try:
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_employees.to_excel(writer, sheet_name='员工', index=False)
        df_scores.to_excel(writer, sheet_name='成绩', index=False)
    print(f"\n已保存 Excel 文件: {excel_path}")

    # 读取 Excel
    df_excel = pd.read_excel(excel_path, sheet_name='员工')
    print(f"\n读取 Excel '员工' sheet:\n{df_excel.head(3)}")

    df_excel_scores = pd.read_excel(excel_path, sheet_name='成绩')
    print(f"\n读取 Excel '成绩' sheet:\n{df_excel_scores.head(3)}")

    # 读取所有 sheet
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    print(f"\n所有 sheet 名称: {list(all_sheets.keys())}")

except ImportError:
    print("\n未安装 openpyxl，跳过 Excel 读写演示")
    print("安装命令: pip install openpyxl")

print("\n" + "=" * 60)
print("四、JSON 文件写入与读取")
print("=" * 60)

json_path = OUTPUT_DIR / 'employees.json'
df_employees.to_json(json_path, orient='records', force_ascii=False, indent=2)
print(f"\n已保存 JSON 文件: {json_path}")

df_json = pd.read_json(json_path)
print(f"\n读取 JSON 文件:\n{df_json.head(3)}")

# 不同 orient 格式
json_orient_index = OUTPUT_DIR / 'employees_index.json'
df_employees.to_json(json_orient_index, orient='index', force_ascii=False, indent=2)
df_json_index = pd.read_json(json_orient_index, orient='index')
print(f"\norient='index' 读取:\n{df_json_index.head(3)}")

print("\n" + "=" * 60)
print("五、其他常用读写参数")
print("=" * 60)

# skiprows 跳过行
df_skip = pd.read_csv(csv_path, skiprows=[1, 3], encoding='utf-8-sig')
print(f"\nskiprows=[1,3] 跳过第2、4行:\n{df_skip}")

# dtype 指定列类型
df_dtype = pd.read_csv(csv_path, dtype={'薪资': float, '入职年份': str}, encoding='utf-8-sig')
print(f"\n指定列类型 (dtype):\n{df_dtype.dtypes}")

# na_values 指定缺失值标记
csv_na_path = OUTPUT_DIR / 'data_na.csv'
pd.DataFrame({'A': [1, 2, 'NA', 4], 'B': ['x', 'missing', 'z', 'w']}).to_csv(
    csv_na_path, index=False, encoding='utf-8-sig')
df_na = pd.read_csv(csv_na_path, na_values=['NA', 'missing'], encoding='utf-8-sig')
print(f"\n指定缺失值标记 (na_values):\n{df_na}")

print("\n" + "=" * 60)
print("六、to_csv() 常用参数")
print("=" * 60)

csv_out_path = OUTPUT_DIR / 'output_custom.csv'
df_employees.to_csv(csv_out_path, index=False, encoding='utf-8-sig', columns=['姓名', '薪资'])
print(f"\n只导出部分列: {csv_out_path}")

csv_float_path = OUTPUT_DIR / 'output_float.csv'
pd.DataFrame({'x': [1.23456, 2.34567], 'y': [3.45678, 4.56789]}).to_csv(
    csv_float_path, index=False, float_format='%.2f')
print(f"\n控制浮点格式 (float_format='%.2f'):")
print(pd.read_csv(csv_float_path))

print("\n所有输出文件保存在:", OUTPUT_DIR)
