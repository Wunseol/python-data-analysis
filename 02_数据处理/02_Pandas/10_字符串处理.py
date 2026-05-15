# 数据来源: 脚本内自建示例数据
# 学习场景: Pandas API 速查 — 逐个演示 str 访问器方法的基本用法
# 综合实战场景: 详见 03_数据清洗/5_字符串清洗与正则.py，包含 re 模块、清洗管道和实战案例

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("一、准备示例数据")
print("=" * 60)

df = pd.DataFrame({
    '姓名': ['  张三 ', '李四', '  王五  ', '赵六', '钱七', '孙八', '周九', '吴十'],
    '邮箱': ['zhangsan@163.com', 'LISI@Gmail.com', 'wangwu@qq.com', 'ZHAOLIU@126.COM',
             'qianqi@gmail.com', 'sunba@163.com', 'zhoujiu@qq.com', 'wushi@sina.com'],
    '电话': ['138-0000-1234', '139 1111 2222', '13755556789', '136-0000-1111',
             '135 2222 3333', '13444445555', '133-6666-7777', '13288889999'],
    '地址': ['北京市朝阳区建国路100号', '上海市浦东新区陆家嘴金融中心',
             '广州市天河区体育西路200号', '深圳市南山区科技园',
             '杭州市西湖区文三路300号', '北京市海淀区中关村',
             '上海市徐汇区漕河泾', '广州市白云区机场路'],
    '备注': ['优秀员工,年度奖', '迟到2次,警告', '正常', '优秀员工,季度奖,年度奖',
             '正常', '请假5天', '优秀员工', '迟到1次']
})

print(df.to_string())

print("\n" + "=" * 60)
print("二、str.lower / upper / title")
print("=" * 60)

print("\nstr.lower() 转小写:")
print(df['邮箱'].str.lower())

print("\nstr.upper() 转大写:")
print(df['邮箱'].str.upper())

print("\nstr.title() 首字母大写:")
print(pd.Series(['hello world', 'pandas demo']).str.title())

print("\n" + "=" * 60)
print("三、str.strip / lstrip / rstrip")
print("=" * 60)

print("\n原始姓名列 (含空格):")
print(df['姓名'].tolist())
print(f"含空格的值: '{df['姓名'].iloc[0]}', '{df['姓名'].iloc[2]}'")

print("\nstr.strip() 去除首尾空格:")
df['姓名'] = df['姓名'].str.strip()
print(df['姓名'].tolist())

print("\nstr.lstrip() 去除左侧空格:")
print(pd.Series(['  left  ', '  space  ']).str.lstrip().tolist())

print("\nstr.rstrip() 去除右侧空格:")
print(pd.Series(['  right  ', '  space  ']).str.rstrip().tolist())

print("\n" + "=" * 60)
print("四、str.replace() 替换")
print("=" * 60)

# 基本替换
df['电话_格式化'] = df['电话'].str.replace('-', '').str.replace(' ', '')
print("\n电话号码去除分隔符:")
print(df[['电话', '电话_格式化']].to_string())

# 正则替换
df['邮箱域名'] = df['邮箱'].str.lower().str.replace(r'@.*', '', regex=True)
print("\n提取邮箱用户名 (正则替换):")
print(df[['邮箱', '邮箱域名']].to_string())

# 替换地址中的数字
df['地址_去数字'] = df['地址'].str.replace(r'\d+', 'XX', regex=True)
print("\n地址中数字替换为XX:")
print(df[['地址', '地址_去数字']].to_string())

print("\n" + "=" * 60)
print("五、str.extract() 正则提取")
print("=" * 60)

# 提取邮箱域名
df['邮箱后缀'] = df['邮箱'].str.extract(r'@([\w.]+)')
print("\n提取邮箱域名后缀:")
print(df[['邮箱', '邮箱后缀']].to_string())

# 提取多个分组
phone_extract = df['电话'].str.replace(r'[-\s]', '', regex=True).str.extract(r'(\d{3})(\d{4})(\d{4})')
phone_extract.columns = ['区号段', '中间段', '尾号段']
print("\n提取电话号码各段:")
print(phone_extract.to_string())

# 提取地址中的城市
df['城市'] = df['地址'].str.extract(r'^(.+?市)')
print("\n提取城市:")
print(df[['地址', '城市']].to_string())

# extractall 提取所有匹配
text_series = pd.Series(['abc123def456', 'xyz789'])
print("\nextractall 提取所有数字:")
print(text_series.str.extractall(r'(\d+)'))

print("\n" + "=" * 60)
print("六、str.contains() 包含检测")
print("=" * 60)

# 检测是否包含某字符串
mask_163 = df['邮箱'].str.contains('163', case=False)
print("\n包含 '163' 的邮箱:")
print(df[mask_163]['邮箱'].tolist())

# 使用正则表达式
mask_gmail = df['邮箱'].str.contains(r'gmail|qq', case=False)
print("\n包含 gmail 或 qq 的邮箱:")
print(df[mask_gmail]['邮箱'].tolist())

# startswith / endswith
mask_start = df['姓名'].str.startswith('张')
print("\n姓张的人:")
print(df[mask_start]['姓名'].tolist())

mask_end = df['邮箱'].str.lower().str.endswith('.com')
print("\n以 .com 结尾的邮箱:")
print(df[mask_end]['邮箱'].tolist())

print("\n" + "=" * 60)
print("七、str.split() 分割")
print("=" * 60)

# 基本分割
df['备注列表'] = df['备注'].str.split(',')
print("\nstr.split(',') 分割备注:")
print(df[['备注', '备注列表']].to_string())

# expand=True 展开为多列
df_split = df['备注'].str.split(',', expand=True)
df_split.columns = [f'备注_{i+1}' for i in range(df_split.shape[1])]
print("\nexpand=True 展开为多列:")
print(df_split.to_string())

# 分割电话号码
df_phone_split = df['电话'].str.replace(r'[-\s]', '', regex=True).str.split(r'(\d{4})', expand=True)
print("\n电话号码分割:")
print(df_phone_split.to_string())

# n 参数限制分割次数
df_split_n = df['备注'].str.split(',', n=1, expand=True)
df_split_n.columns = ['主要备注', '其他备注']
print("\nn=1 限制分割次数:")
print(df_split_n.to_string())

# rsplit 从右侧分割
s = pd.Series(['a-b-c-d'])
print(f"\nsplit('-', n=2): {s.str.split('-', n=2).tolist()}")
print(f"rsplit('-', n=2): {s.str.rsplit('-', n=2).tolist()}")

print("\n" + "=" * 60)
print("八、str.len() 长度")
print("=" * 60)

df['姓名长度'] = df['姓名'].str.len()
print("\n姓名长度:")
print(df[['姓名', '姓名长度']].to_string())

# 筛选长度
long_addr = df[df['地址'].str.len() > 12]
print("\n地址长度>12的记录:")
print(long_addr[['地址']].to_string())

print("\n" + "=" * 60)
print("九、其他 str 方法")
print("=" * 60)

# str.count() 计数
df['逗号数量'] = df['备注'].str.count(',')
print("\n备注中逗号数量:")
print(df[['备注', '逗号数量']].to_string())

# str.find() / str.index() 查找位置
print("\nstr.find('区') 查找区字位置:")
print(df['地址'].str.find('区').tolist())

# str.pad() / str.center() / str.zfill() 填充
print("\nstr.zfill(6) 前导零填充:")
print(pd.Series(['1', '23', '456']).str.zfill(6).tolist())

print("\nstr.center(10, '-') 居中填充:")
print(pd.Series(['hi', 'hello']).str.center(10, '-').tolist())

# str.repeat() 重复
print("\nstr.repeat(2) 重复:")
print(pd.Series(['ab', 'cd']).str.repeat(2).tolist())

# str.slice() 切片
print("\nstr.slice(0, 3) 取前3个字符:")
print(df['姓名'].str.slice(0, 3).tolist())

print("\n" + "=" * 60)
print("十、文本清洗管道")
print("=" * 60)

raw_text = pd.Series([
    '  Hello, World!  ',
    'Python数据分析  ',
    '  Pandas  STR  方法  ',
    '  数据 清洗  Pipeline  '
])

print("\n原始文本:")
print(raw_text.tolist())

cleaned = (raw_text
           .str.strip()
           .str.lower()
           .str.replace(r'[^\w\u4e00-\u9fff]', ' ', regex=True)
           .str.replace(r'\s+', ' ', regex=True)
           .str.strip())

print("\n清洗后文本 (strip → lower → 去标点 → 合并空格 → strip):")
print(cleaned.tolist())

print("\n" + "=" * 60)
print("十一、综合示例: 清洗邮箱和电话")
print("=" * 60)

df_result = pd.DataFrame({
    '原始邮箱': df['邮箱'],
    '原始电话': df['电话']
})

df_result['邮箱_小写'] = df_result['原始邮箱'].str.lower()
df_result['邮箱_域名'] = df_result['原始邮箱'].str.extract(r'@([\w.]+)')

phone_clean = df_result['原始电话'].str.replace(r'[-\s]', '', regex=True)
df_result['电话_格式化'] = phone_clean.str.replace(r'(\d{3})(\d{4})(\d{4})', r'\1-\2-\3', regex=True)

print("\n清洗结果:")
print(df_result.to_string())
