# 导入requests库，用于发送HTTP请求
import requests

# 定义商品URL，用于爬取Amazon商品页面
url="https://www.sonkwo.cn/store/search"

# 尝试发送HTTP请求并获取网页内容
try:
    # 设置请求头，模拟浏览器访问，防止被网站拒绝
    kv={'user-agent':'Mozilla/5.0'}
    # 发送GET请求获取网页内容
    r=requests.get(url,headers=kv)
    # 检查请求状态，如果发生错误则抛出异常
    r.raise_for_status()
    # 设置正确的编码方式，防止中文乱码
    r.encoding=r.apparent_encoding
    # 打印网页内容
    print(r.text)
# 异常处理，捕获所有可能的错误并打印提示信息
except:
        print("error")


# 该Python函数功能如下：

# 使用requests库发送HTTP GET请求访问指定Amazon商品页面。
# 通过设置请求头模拟浏览器行为，避免请求被服务器拒绝。
# 检查请求状态，确保请求成功；若请求失败，则捕获异常并输出错误信息。
# 为了解决可能出现的中文乱码问题，根据响应内容设定正确的字符编码。
# 最终打印出获取到的网页源代码。


