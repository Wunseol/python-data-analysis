# 导入所需模块
import requests
import time
import re
import codecs
from bs4 import BeautifulSoup

# 定义一个字典，用来模拟浏览器头部信息
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0'
}

# 使用codecs库以追加模式打开一个文件，用于存储爬取的数据，如果 D: 盘不存在该文件，codecs.open() 会尝试创建这个文件。
f = codecs.open('D:\sanguo.txt', 'a+', encoding='utf-8')


def get_info(url):
    """
    获取指定URL页面上的信息
    :param url: 页面URL
    """
    # 发送GET请求获取网页内容
    res = requests.get(url, headers=header)
    # 检查请求是否成功
    if res.status_code == 200:
        # 设置正确的编码方式以解析响应内容
        res.encoding = res.apparent_encoding
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(res.text, 'html.parser')
        # 选择特定的元素集合
        contents = soup.select('div:nth-child(2) > div.c > ul > li > a')
        # 遍历并打印元素集合中的每个元素
        for content in contents:
            print(content)
        # 再次遍历元素集合，打印元素的文本内容并写入文件
        for content in contents:
            print(content.string)
            f.write(content.string + '\n')
    else:
        # 如果请求不成功，不执行任何操作
        pass


# 程序入口
if __name__ == '__main__':
    # 定义目标URL
    url = "http://www.kulemi.com/zt/127/"
    # 调用函数获取URL上的信息
    get_info(url)
    # 关闭文件
    f.close()


# 该Python程序用于从指定网站抓取数据并保存到本地文件中。主要功能如下：

# 使用requests库发送HTTP GET请求获取网页内容。
# 利用BeautifulSoup解析HTML文档，并提取特定元素（位于div:nth-child(2) > div.c > ul > li > a路径下的标签）。
# 打印提取的内容，并将文本信息追加写入到D:\sanguo.txt文件中。
# 请求成功时执行上述操作，否则不做处理。


