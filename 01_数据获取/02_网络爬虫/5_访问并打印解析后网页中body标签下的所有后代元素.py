# 导入BeautifulSoup库和requests库，用于网页请求和HTML解析
from bs4 import BeautifulSoup
import requests

# 使用requests库获取京东首页的HTML源码
r = requests.get('https://www.jd.com/')

# 使用BeautifulSoup解析获取的HTML文本
soup = BeautifulSoup(r.text, 'html.parser')

# 访问并打印解析后网页中body标签下的所有后代元素（包括子元素的子元素等）
a = soup.body.descendants
# 你可以将生成器对象转换为列表并打印，以便查看所有元素：
print(a)
# 这样会将所有后代元素以列表形式打印出来。
print(list(a))

