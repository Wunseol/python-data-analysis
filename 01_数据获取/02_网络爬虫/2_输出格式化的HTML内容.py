import requests     # 导入requests库，用于发送HTTP请求
from bs4 import BeautifulSoup       # 导入BeautifulSoup库，用于解析HTML和XML文档



# 使用GET方法请求网站的HTML内容
r = requests.get("https://www.sonkwo.cn/store/search")

# 访问响应的文本内容，存储知乎网站的HTML代码
r.text

# 将知乎网站的HTML代码赋值给变量demo
demo = r.text


# 使用BeautifulSoup解析demo中的HTML内容，并指定解析器为'html.parser'
soup = BeautifulSoup(demo, 'html.parser')

# 使用prettify方法格式化输出解析后的HTML内容，便于阅读和调试
print(soup.prettify())


# 该Python函数实现以下功能：

# 使用requests库的get方法访问知乎网站并获取其HTML内容。
# 将获取到的HTML内容存储在变量demo中。
# 利用BeautifulSoup库解析demo中的HTML，选择的解析器为html.parser。
# 通过prettify方法美化输出格式化的HTML内容，增强可读性。
