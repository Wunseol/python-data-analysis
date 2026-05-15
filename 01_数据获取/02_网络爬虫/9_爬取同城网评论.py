# 导入必要的库
import requests
import urllib.request
import json
import time

# 设置请求头，模拟浏览器访问
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
}

def getdate():
    """
    爬取某网站上的点评数据。

    该函数通过循环请求特定城市的点评页面，并将每条评论解析后写入本地文件。
    """
    # 设置城市代码，这里以"5162"为例
    cityname = "5162"
    # 初始化评论编号
    j = 1
    # 循环请求10页点评数据
    for i in range(1, 11):
        # 每次请求间隔2秒，防止过于频繁的访问
        time.sleep(2)
        # 构造请求URL
        geturl = "https://www.ly.com/scenery/AjaxHelper/DianPingAjax.aspx?action=GetDianPingList&sid=%s&page="%cityname + str(i) + "&pageSize=10"
        # 发送请求并获取响应
        request = requests.get(url=geturl, headers=headers)
        # 解析响应为JSON对象
        html = json.loads(request.text)
        # 打印当前页数
        print('正在爬取第' + str(i) + '页')
        # 解析数据
        datas = html['dpList']
        # 将评论数据写入文件
        with open("tongcheng.txt", "a", newline='', encoding='utf-8') as f:
            for k in datas:
                # 写入评论，每条评论前加编号
                f.write('(' + str(j) + ')' + k['dpContent'])
                f.write("\n")
                # 编号递增
                j += 1

# 程序入口
if __name__ == '__main__':
    getdate()



# 该Python函数用于爬取某网站上的点评数据。
# 设置了请求头headers以模拟浏览器访问。
# getdate()函数中，循环10次，每次请求一个特定城市的点评页面。
# 使用requests.get获取网页内容，并用json.loads解析为Python对象。
# 提取并打印当前页数，随后处理数据，将每条评论写入本地文本文件"tongcheng.txt"中。
# 每条评论前加编号，逐条写入文件。

# 注：该程序应用到本人pythonweb项目爬取同城网评论，仅供学习参考。