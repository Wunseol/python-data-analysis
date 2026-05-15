import requests
import json
import time

# 设置请求头，模拟浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
}

# 携程网评论数据抓取的URL
posturl = "https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList?_fxpcqlniredt=09031028411306444964"

def getdata():
    """
    爬取携程网评论数据并保存到文件

    本函数通过循环发送POST请求，抓取携程网的评论数据。
    每次请求获取一页评论，共抓取50页。抓取的内容包括评论的详细信息，
    如评论文本等。抓取的数据被保存到名为'xiecheng.txt'的文件中。
    """
    # 初始化评论编号
    j = 1
    # 循环抓取50页评论
    for i in range(1, 51):
        # 构造请求参数
        request = {
            'arg': {'channelType': '2',
                    'collapseType': '0',
                    'commentTagId': '0',
                    'pageIndex': str(i),
                    'pageSize': '10',
                    'poiId': '10558614',
                    'sortType': '3',
                    'sourceType': '1',
                    'starType': '0'},

            'head': {'auth': "",
                     'cid': "09031028411306444964",
                     'ctok': "",
                     'cver': "1.0",
                     'extension': [],
                     'lang': "01",
                     'sid': "8888",
                     'syscode': "09",
                     'xsid': ""}
        }

        # 休眠3秒，避免请求过于频繁
        time.sleep(3)
        # 发送POST请求
        html = requests.post(posturl, data=json.dumps(request), headers=headers)
        # 解析响应
        html1 = json.loads(html.text)
        # 打印当前抓取的页数
        print('正在爬取第'+str(i)+'页')
        # 获取评论项
        items = html1['result']['items']
        # 保存文件
        with open("xiecheng.txt", "a", newline='', encoding='utf-8') as f:
            for k in items:
                # 写入评论内容
                f.write('(' + str(j) + ')' + k['content'])
                f.write("\n")
                # 更新评论编号
                j += 1

if __name__ == '__main__':
    getdata()

# 该Python函数getdata()用于爬取携程网上特定poiId下的评论数据。具体功能如下：

# 循环请求指定URL，获取共50页的评论数据；
# 每次请求间休眠3秒以避免被封IP；
# 解析返回的JSON数据，提取评论内容；
# 将评论逐条写入本地文本文件"xiecheng.txt"中。

