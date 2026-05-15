# 数据来源: 脚本内自建示例数据
# 本脚本演示 Pandas 数据变换与重塑操作

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

df_wide = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '语文': [85, 92, 78],
    '数学': [90, 76, 95],
    '英语': [78, 88, 82]
})

print("宽格式数据:")
print(df_wide)

print("\n" + "=" * 60)
print("二、melt() 宽转长")
print("=" * 60)

# 基本 melt
df_long = pd.melt(df_wide, id_vars=['姓名'])
print("\nmelt 基本转换 (宽→长):")
print(df_long)

# 指定 variable 和 value 列名
df_long_named = pd.melt(df_wide, id_vars=['姓名'],
                         var_name='科目', value_name='成绩')
print("\n指定列名:")
print(df_long_named)

# 只选择部分变量列
df_long_subset = pd.melt(df_wide, id_vars=['姓名'],
                          value_vars=['语文', '数学'],
                          var_name='科目', value_name='成绩')
print("\n只选择部分变量列:")
print(df_long_subset)

print("\n" + "=" * 60)
print("三、pivot() 长转宽")
print("=" * 60)

# pivot 将长格式转回宽格式
df_wide_restored = df_long_named.pivot(index='姓名', columns='科目', values='成绩')
print("\npivot 长转宽:")
print(df_wide_restored)

# reset_index 使索引变为列
df_wide_flat = df_wide_restored.reset_index()
df_wide_flat.columns.name = None
print("\nreset_index 后:")
print(df_wide_flat)

# 多值 pivot
df_multi = pd.DataFrame({
    '姓名': ['张三', '张三', '张三', '李四', '李四', '李四'],
    '科目': ['语文', '语文', '数学', '语文', '语文', '数学'],
    '类型': ['期中', '期末', '期中', '期中', '期末', '期中'],
    '成绩': [80, 85, 88, 90, 92, 75]
})
print("\n多值长格式数据:")
print(df_multi)

df_pivot_multi = df_multi.pivot(index='姓名', columns='科目', values='成绩')
print("\npivot 多值 (如有重复会报错, 需用 pivot_table):")
print(df_pivot_multi)

print("\n" + "=" * 60)
print("四、stack() 与 unstack()")
print("=" * 60)

# 准备多层索引数据
df_multi_idx = pd.DataFrame({
    '期中': [85, 92, 78],
    '期末': [90, 88, 82]
}, index=pd.Index(['张三', '李四', '王五'], name='姓名'))
df_multi_idx.columns.name = '考试类型'

print("原始数据:")
print(df_multi_idx)

# stack 列转行
df_stacked = df_multi_idx.stack()
print("\nstack() 列转行 (Series):")
print(df_stacked)

# unstack 行转列
df_unstacked = df_stacked.unstack()
print("\nunstack() 行转列 (还原):")
print(df_unstacked)

# 多层索引的 stack/unstack
df_hierarchical = pd.DataFrame({
    '语文': [85, 78, 92, 88],
    '数学': [90, 95, 76, 82]
}, index=pd.MultiIndex.from_tuples(
    [('张三', '期中'), ('张三', '期末'), ('李四', '期中'), ('李四', '期末')],
    names=['姓名', '考试类型']
))

print("\n多层索引数据:")
print(df_hierarchical)

# unstack 不同层级
print("\nunstack(level='考试类型'):")
print(df_hierarchical.unstack(level='考试类型'))

print("\nunstack(level='姓名'):")
print(df_hierarchical.unstack(level='姓名'))

# stack 不同层级
df_cols_multi = df_hierarchical.unstack(level='考试类型')
print("\nstack(level='考试类型') 还原:")
print(df_cols_multi.stack(level='考试类型'))

print("\n" + "=" * 60)
print("五、wide_to_long() 宽转长")
print("=" * 60)

df_wide_panel = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '成绩_期中_语文': [80, 90, 75],
    '成绩_期中_数学': [85, 78, 92],
    '成绩_期末_语文': [88, 92, 80],
    '成绩_期末_数学': [90, 80, 95]
})

print("宽格式面板数据:")
print(df_wide_panel)

df_panel_long = pd.wide_to_long(
    df_wide_panel,
    stubnames='成绩',
    i='姓名',
    j='考试_科目',
    sep='_',
    suffix='.+'
).reset_index()

print("\nwide_to_long 转换:")
print(df_panel_long)

# 手动拆分 j 列
df_panel_long[['考试类型', '科目']] = df_panel_long['考试_科目'].str.split('_', expand=True)
df_panel_long = df_panel_long.drop(columns=['考试_科目'])
print("\n拆分后:")
print(df_panel_long)

print("\n" + "=" * 60)
print("六、分类数据 (Categorical) 操作")
print("=" * 60)

# 创建分类数据
df_cat = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '绩效': pd.Categorical(['A', 'B', 'C', 'A', 'B'], categories=['C', 'B', 'A'], ordered=True),
    '部门': pd.Categorical(['技术部', '市场部', '技术部', '财务部', '市场部'])
})

print("分类数据:")
print(df_cat)
print(f"\n绩效列类型: {df_cat['绩效'].dtype}")
print(f"绩效类别: {df_cat['绩效'].cat.categories.tolist()}")
print(f"绩效是否有序: {df_cat['绩效'].cat.ordered}")

# 分类操作
print(f"\n重命名类别:")
df_cat['绩效'] = df_cat['绩效'].cat.rename_categories({'C': '需改进', 'B': '良好', 'A': '优秀'})
print(df_cat['绩效'])

# 添加新类别
df_cat['部门'] = df_cat['部门'].cat.add_categories(['人事部'])
print(f"\n添加类别后: {df_cat['部门'].cat.categories.tolist()}")

# 删除类别
df_cat['部门'] = df_cat['部门'].cat.remove_categories(['人事部'])
print(f"删除类别后: {df_cat['部门'].cat.categories.tolist()}")

# 有序分类排序
print("\n按有序分类排序:")
print(df_cat.sort_values('绩效')[['姓名', '绩效']].to_string())

# 分类编码
print(f"\n分类编码 (codes): {df_cat['绩效'].cat.codes.tolist()}")
print(f"对应关系: {dict(zip(df_cat['绩效'].cat.codes, df_cat['绩效']))}")

print("\n" + "=" * 60)
print("七、cut() 等距分箱")
print("=" * 60)

ages = pd.Series([22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 28, 33, 48])
print(f"原始年龄数据: {ages.tolist()}")

# 等距分箱
bins_3 = pd.cut(ages, bins=3)
print(f"\ncut 3个等距箱:")
print(bins_3)
print(f"箱计数:\n{bins_3.value_counts().sort_index()}")

# 自定义分箱边界
bins_custom = pd.cut(ages, bins=[0, 25, 35, 50, 100],
                     labels=['青年', '壮年', '中年', '老年'])
print(f"\n自定义分箱:")
print(bins_custom)
print(f"箱计数:\n{bins_custom.value_counts().sort_index()}")

# right=False 左闭右开
bins_left = pd.cut(ages, bins=[0, 25, 35, 50, 100], right=False)
print(f"\nright=False 左闭右开:")
print(bins_left)

# include_lowest=True
bins_lowest = pd.cut(ages, bins=[22, 35, 50, 100], include_lowest=True)
print(f"\ninclude_lowest=True:")
print(bins_lowest)

print("\n" + "=" * 60)
print("八、qcut() 等频分箱")
print("=" * 60)

scores = pd.Series([56, 62, 71, 75, 78, 82, 85, 88, 91, 95, 98, 45, 67, 73, 80])
print(f"原始成绩数据: {scores.tolist()}")

# 等频分箱 (每箱样本数大致相等)
qcut_4 = pd.qcut(scores, q=4)
print(f"\nqcut 4等分:")
print(qcut_4)
print(f"箱计数:\n{qcut_4.value_counts().sort_index()}")

# 自定义分位数
qcut_custom = pd.qcut(scores, q=[0, 0.3, 0.7, 1.0], labels=['低', '中', '高'])
print(f"\n自定义分位数 (30%-70%):")
print(qcut_custom)
print(f"箱计数:\n{qcut_custom.value_counts().sort_index()}")

# cut vs qcut 对比
print("\ncut vs qcut 对比:")
print(f"cut 3等距: {pd.cut(scores, bins=3).value_counts().sort_index().tolist()}")
print(f"qcut 3等频: {pd.qcut(scores, q=3).value_counts().sort_index().tolist()}")

print("\n" + "=" * 60)
print("九、综合示例: 成绩数据重塑")
print("=" * 60)

df_exam = pd.DataFrame({
    '姓名': ['张三'] * 3 + ['李四'] * 3 + ['王五'] * 3,
    '科目': ['语文', '数学', '英语'] * 3,
    '期中': [80, 90, 78, 92, 76, 88, 78, 95, 82],
    '期末': [85, 88, 80, 90, 80, 85, 82, 92, 86]
})

print("原始成绩数据 (半长格式):")
print(df_exam)

# melt 转为完全长格式
df_full_long = pd.melt(df_exam, id_vars=['姓名', '科目'],
                        var_name='考试类型', value_name='成绩')
print("\nmelt 转完全长格式:")
print(df_full_long.to_string())

# pivot 转回半长格式
df_semi_wide = df_full_long.pivot(index=['姓名', '科目'], columns='考试类型', values='成绩').reset_index()
df_semi_wide.columns.name = None
print("\npivot 转回半长格式:")
print(df_semi_wide.to_string())

# pivot_table 生成汇总
df_summary = pd.pivot_table(df_full_long, values='成绩',
                             index='姓名', columns='科目',
                             aggfunc='mean')
print("\npivot_table 汇总 (平均成绩):")
print(df_summary.round(2))

# 添加分箱
df_full_long['成绩等级'] = pd.cut(df_full_long['成绩'],
                                   bins=[0, 60, 70, 80, 90, 100],
                                   labels=['不及格', '及格', '中等', '良好', '优秀'])
print("\n添加成绩等级:")
print(df_full_long.to_string())

# 统计各等级人数
grade_count = pd.crosstab(df_full_long['姓名'], df_full_long['成绩等级'])
print("\n各人成绩等级分布:")
print(grade_count)
