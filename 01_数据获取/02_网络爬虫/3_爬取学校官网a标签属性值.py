# 导入urllib.request模块，用于打开和读取网页
import urllib.request

# 导入BeautifulSoup库，用于解析HTML内容
from bs4 import BeautifulSoup

# 打开指定URL，读取并解码网页内容
html = urllib.request.urlopen("https://www.sspu.edu.cn/").read().decode("utf-8")

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html, features='html.parser')

# 查找页面中所有的<a>标签
tags = soup.find_all('a')

# 遍历所有<a>标签
for tag in tags:
    # 打印每个<a>标签的href属性值，去除首尾空白
    print(str(tag.get('href')).strip())


# 该Python代码从指定网站抓取网页内容，并使用BeautifulSoup解析。主要功能如下：
# 从上海第二工业大学官网获取HTML源码。
# 解析HTML，查找所有<a>标签。
# 打印每个<a>标签的href属性值（已去除首尾空白）。

