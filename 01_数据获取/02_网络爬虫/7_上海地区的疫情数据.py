# 导入requests库，用于发送HTTP请求
import requests

# 使用GET方法请求上海地区的疫情数据
# 参数from=mola-virus表示请求来源，stage=publish表示发布阶段，target=trend表示目标是趋势数据
# isCaseIn=1表示包含确诊病例数据，area=%E4%B8%8A%E6%B5%B7表示上海地区
r =  requests.get("https://voice.baidu.com/newpneumonia/getv2?from=mola-virus&stage=publish&target=trend&isCaseIn=1&area=%E4%B8%8A%E6%B5%B7")

# 打印请求状态码，200表示请求成功
print(r.status_code)

# 设置请求结果的编码为utf-8，确保中文字符正确处理
r.encoding = 'utf-8'

# 打印请求结果的文本内容
print(r.text)


# 该Python函数执行以下操作：

# 使用requests库通过GET方法从百度服务器获取上海地区的疫情趋势数据。
# 请求特定参数以指定数据来源、发布阶段、目标类型、数据包含内容及地区。
# 打印HTTP请求的状态码，确认请求是否成功（200代表成功）。
# 修改响应编码格式为UTF-8保证中文字符正确处理，随后打印返回的文本内容。

