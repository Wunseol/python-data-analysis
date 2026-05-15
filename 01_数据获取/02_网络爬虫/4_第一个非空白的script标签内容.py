import requests
from bs4 import BeautifulSoup

# 发送GET请求到淘宝网站的首页，并获取响应对象r
r = requests.get('https://www.taobao.com/')
# 将响应内容转换为字符串存储在html变量中
html = r.text

# 使用BeautifulSoup解析HTML字符串，'html.parser'指定解析器类型
soup = BeautifulSoup(html, 'html.parser')

# 获取soup对象中body标签的所有后代元素
a = soup.body.descendants

# print(type(a))
# print(a)
# print('++++++++++++++++++++++++\n')

# 初始化一个空字符串，用于存储找到的非空白元素
res = ''
# 遍历body的所有后代元素
for i in a:
    # 检查元素是否为非空白字符串
    if str(i).strip():
        # 打印找到的非空白元素及其位置信息
        print('+++\n', i)
        # 将找到的非空白元素存储在res变量中
        res = i
        # 找到第一个非空白元素后停止循环
        break
# 打印三个换行，用于分隔元素
print('\n\n\n')
# 打印最终找到的非空白元素及其位置信息
print('++++++++++++++++++++++++\n', res, '\n++++++++++++++++++++++++')
print('\n\n\n')


# 该Python代码从淘宝网站获取HTML内容，并使用BeautifulSoup解析。主要功能包括：
# 1. 发送HTTP请求至淘宝首页并获取响应。
# 2. 使用BeautifulSoup解析HTML文本。
# 3. 遍历<body>标签下的所有后代元素，查找非空白字符串，并打印第一个非空白元素及其位置信息。

# 淘宝网站获取HTML内容，第一个非空白的script标签内容

