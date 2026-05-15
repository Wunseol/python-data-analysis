# 数据来源: 自建 DataFrame 演示数据

import pandas as pd
import numpy as np
import re
from pathlib import Path

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 100)

# ============================================================
# 一、构建演示数据
# ============================================================
data = {
    "姓名": ["  张三 ", "李 四", "王  五  ", "赵六", "钱 七 "],
    "电话": ["138-1234-5678", "139 5678 9012", "13612345678", "137-9999-8888", "135 6666 7777"],
    "邮箱": ["zhangsan@163.com", "LI_SI@gmail.com", "wangwu@qq.com  ", "ZHAOLIU@126.COM", "qianqi@163.com"],
    "地址": ["北京市朝阳区XX路123号", "上海市浦东新区YY路456号", "广州市天河区ZZ路789号", "深圳市南山区AA路101号", "杭州市西湖区BB路202号"],
    "金额": ["￥1,234.56", "￥5,678.90", "2,345.67", "￥890.12", "3,456,789.00"],
    "身份证": ["110101199001011234", "310101198512052345", "440101199203033456", "440301199005055678", "330101199107077890"],
    "备注": ["正常##用户", "VIP**客户", "普通@@用户", "钻石!!客户", "正常//用户"],
}
df = pd.DataFrame(data)
print("=== 原始数据 ===")
print(df)
print()

# ============================================================
# 二、str 访问器方法
# ============================================================

# 2.1 strip / lstrip / rstrip: 去除空白字符
print("=== str.strip() 去除首尾空白 ===")
df_strip = df.copy()
df_strip["姓名"] = df_strip["姓名"].str.strip()
print(df_strip["姓名"])
print()

# 2.2 str.replace: 替换字符串
print("=== str.replace() 替换电话中的分隔符 ===")
df_phone = df.copy()
df_phone["电话_clean"] = df_phone["电话"].str.replace("-", "", regex=False)
df_phone["电话_clean"] = df_phone["电话_clean"].str.replace(" ", "", regex=False)
print(df_phone[["电话", "电话_clean"]])
print()

# 2.3 str.lower / str.upper / str.title: 大小写转换
print("=== str.lower() 邮箱转小写 ===")
df_email = df.copy()
df_email["邮箱_clean"] = df_email["邮箱"].str.strip().str.lower()
print(df_email[["邮箱", "邮箱_clean"]])
print()

# 2.4 str.contains: 包含检测
print("=== str.contains() 检测邮箱域名 ===")
is_163 = df["邮箱"].str.contains("163", regex=False)
print(f"163邮箱用户:")
print(df[is_163][["姓名", "邮箱"]])
print()

# 2.5 str.startswith / str.endswith
print("=== str.startswith() 身份证前缀 ===")
beijing = df["身份证"].str.startswith("110")
print(f"北京用户:")
print(df[beijing][["姓名", "身份证"]])
print()

# 2.6 str.len: 字符串长度
print("=== str.len() 字符串长度 ===")
print(df["姓名"].str.strip().str.len())
print()

# 2.7 str.split: 分割字符串
print("=== str.split() 分割地址 ===")
df_split = df.copy()
df_split["省份"] = df_split["地址"].str.split("市", n=1).str[0] + "市"
df_split["区域"] = df_split["地址"].str.split("区", n=1).str[0].str.split("市", n=1).str[1] + "区"
print(df_split[["地址", "省份", "区域"]])
print()

# 2.8 str.slice: 切片
print("=== str.slice() 身份证提取出生日期 ===")
df_id = df.copy()
df_id["出生日期"] = df_id["身份证"].str.slice(6, 14)
print(df_id[["姓名", "身份证", "出生日期"]])
print()

# ============================================================
# 三、正则表达式 — re 模块
# ============================================================

# 3.1 re.sub: 替换特殊字符
print("=== re.sub() 去除备注中的特殊字符 ===")
df_remark = df.copy()
df_remark["备注_clean"] = df_remark["备注"].apply(lambda x: re.sub(r"[#*@!/]", "", x))
print(df_remark[["备注", "备注_clean"]])
print()

# 3.2 re.findall: 提取所有匹配
print("=== re.findall() 从地址中提取路名和门牌号 ===")
for addr in df["地址"]:
    match = re.findall(r"(.+路)(\d+号)", addr)
    if match:
        road, number = match[0]
        print(f"  {addr} -> 路: {road}, 号: {number}")
print()

# 3.3 re.search: 搜索匹配
print("=== re.search() 验证邮箱格式 ===")
email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
for email in df["邮箱"]:
    email_clean = email.strip()
    is_valid = bool(re.search(email_pattern, email_clean))
    print(f"  {email_clean}: {'有效' if is_valid else '无效'}")
print()

# 3.4 re.match: 从头匹配
print("=== re.match() 验证手机号格式 ===")
phone_pattern = r"^1[3-9]\d{9}$"
for phone in df["电话"]:
    phone_clean = re.sub(r"[-\s]", "", phone)
    is_valid = bool(re.match(phone_pattern, phone_clean))
    print(f"  {phone} -> {phone_clean}: {'有效' if is_valid else '无效'}")
print()

# ============================================================
# 四、pandas 中的正则操作
# ============================================================

# 4.1 str.replace + 正则: 清洗金额列
print("=== str.replace(regex=True) 清洗金额 ===")
df_amount = df.copy()
df_amount["金额_clean"] = df_amount["金额"].str.replace(r"[￥,]", "", regex=True)
df_amount["金额_numeric"] = pd.to_numeric(df_amount["金额_clean"], errors="coerce")
print(df_amount[["金额", "金额_clean", "金额_numeric"]])
print()

# 4.2 str.extract: 提取匹配组
print("=== str.extract() 提取邮箱用户名和域名 ===")
df_extract = df.copy()
df_extract["邮箱用户名"] = df_extract["邮箱"].str.strip().str.extract(r"^([a-zA-Z0-9._%+-]+)@")
df_extract["邮箱域名"] = df_extract["邮箱"].str.strip().str.extract(r"@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
print(df_extract[["邮箱", "邮箱用户名", "邮箱域名"]])
print()

# 4.3 str.extractall: 提取所有匹配
print("=== str.extractall() 提取地址中的所有数字 ===")
numbers = df["地址"].str.extractall(r"(\d+)")
print(numbers)
print()

# 4.4 str.findall: 查找所有匹配
print("=== str.findall() 查找备注中的中文字符 ===")
chinese_chars = df["备注"].str.findall(r"[\u4e00-\u9fff]+")
print(chinese_chars)
print()

# ============================================================
# 五、标准化格式
# ============================================================

# 5.1 电话号码标准化
print("=== 电话号码标准化 ===")
df_std = df.copy()
df_std["电话_clean"] = df_std["电话"].str.replace(r"[-\s]", "", regex=True)
df_std["电话_formatted"] = df_std["电话_clean"].str.replace(
    r"(\d{3})(\d{4})(\d{4})", r"\1-\2-\3", regex=True
)
print(df_std[["电话", "电话_clean", "电话_formatted"]])
print()

# 5.2 身份证信息提取
print("=== 身份证信息提取 ===")
df_id_info = df.copy()
df_id_info["省份码"] = df_id_info["身份证"].str.slice(0, 2)
df_id_info["出生年"] = df_id_info["身份证"].str.slice(6, 10)
df_id_info["出生月"] = df_id_info["身份证"].str.slice(10, 12)
df_id_info["出生日"] = df_id_info["身份证"].str.slice(12, 14)
df_id_info["性别"] = df_id_info["身份证"].str.slice(-2, -1).astype(int).apply(
    lambda x: "男" if x % 2 == 1 else "女"
)
print(df_id_info[["姓名", "省份码", "出生年", "出生月", "出生日", "性别"]])
print()

# 5.3 邮箱标准化
print("=== 邮箱标准化 ===")
df_email_std = df.copy()
df_email_std["邮箱_clean"] = (
    df_email_std["邮箱"]
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "", regex=True)
)
print(df_email_std[["邮箱", "邮箱_clean"]])
print()

# ============================================================
# 六、构建字符串清洗管道
# ============================================================

print("=== 字符串清洗管道 ===")

def clean_pipeline(df_input):
    df_clean = df_input.copy()

    # 步骤1: 去除首尾空白
    str_cols = ["姓名", "邮箱", "备注"]
    for col in str_cols:
        df_clean[col] = df_clean[col].str.strip()

    # 步骤2: 去除多余内部空白
    df_clean["姓名"] = df_clean["姓名"].str.replace(r"\s+", "", regex=True)

    # 步骤3: 标准化电话号码
    df_clean["电话"] = df_clean["电话"].str.replace(r"[-\s]", "", regex=True)

    # 步骤4: 邮箱转小写
    df_clean["邮箱"] = df_clean["邮箱"].str.lower()

    # 步骤5: 清洗金额
    df_clean["金额"] = df_clean["金额"].str.replace(r"[￥,]", "", regex=True)
    df_clean["金额"] = pd.to_numeric(df_clean["金额"], errors="coerce")

    # 步骤6: 去除备注中的特殊字符
    df_clean["备注"] = df_clean["备注"].str.replace(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", regex=True)

    return df_clean

df_pipeline_result = clean_pipeline(df)
print("清洗后数据:")
print(df_pipeline_result)
print()
print("清洗后数据类型:")
print(df_pipeline_result.dtypes)
