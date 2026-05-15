# 数据来源: JSONPlaceholder (https://jsonplaceholder.typicode.com)

import requests
from urllib.parse import urlencode, urlparse, parse_qs
from pathlib import Path

BASE_URL = "https://jsonplaceholder.typicode.com"

MOCK_POSTS = [
    {"userId": 1, "id": 1, "title": "模拟文章1", "body": "内容1"},
    {"userId": 1, "id": 2, "title": "模拟文章2", "body": "内容2"},
    {"userId": 2, "id": 3, "title": "模拟文章3", "body": "内容3"},
]


def fetch_with_fallback(url, params=None, headers=None, timeout=5):
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp
    except (requests.RequestException, Exception):
        return None


def demo_query_params():
    print("=" * 50)
    print("1. Query参数传递")
    print("=" * 50)

    params = {"userId": 1, "_limit": 3}

    print(f"参数字典: {params}")
    print(f"编码后: {urlencode(params)}")

    resp = fetch_with_fallback(f"{BASE_URL}/posts", params=params)
    if resp is not None:
        posts = resp.json()
        print(f"获取到 {len(posts)} 篇文章（userId=1）")
        for p in posts:
            print(f"  ID={p['id']}: {p['title'][:30]}")
    else:
        filtered = [p for p in MOCK_POSTS if p["userId"] == 1][:3]
        print(f"模拟数据: 获取到 {len(filtered)} 篇文章（userId=1）")
        for p in filtered:
            print(f"  ID={p['id']}: {p['title']}")


def demo_url_construction():
    print("\n" + "=" * 50)
    print("2. URL构建与解析")
    print("=" * 50)

    url = f"{BASE_URL}/posts?userId=1&_limit=3"
    parsed = urlparse(url)
    print(f"完整URL: {url}")
    print(f"  协议: {parsed.scheme}")
    print(f"  域名: {parsed.netloc}")
    print(f"  路径: {parsed.path}")
    print(f"  查询参数: {parsed.query}")

    qs = parse_qs(parsed.query)
    print(f"  解析结果: {qs}")

    base = f"{BASE_URL}/posts"
    params = {"userId": 1, "_limit": 3}
    built_url = f"{base}?{urlencode(params)}"
    print(f"\n手动构建URL: {built_url}")

    resp = requests.Request("GET", base, params=params).prepare()
    print(f"requests构建URL: {resp.url}")


def demo_path_params():
    print("\n" + "=" * 50)
    print("3. 路径参数")
    print("=" * 50)

    resource = "posts"
    post_id = 1
    url = f"{BASE_URL}/{resource}/{post_id}"
    print(f"路径参数URL: {url}")

    resp = fetch_with_fallback(url)
    if resp is not None:
        post = resp.json()
        print(f"文章标题: {post['title']}")
    else:
        print(f"模拟数据: {MOCK_POSTS[0]['title']}")

    comment_url = f"{BASE_URL}/{resource}/{post_id}/comments"
    print(f"\n嵌套资源URL: {comment_url}")
    resp = fetch_with_fallback(comment_url)
    if resp is not None:
        comments = resp.json()
        print(f"获取到 {len(comments)} 条评论")
    else:
        print("模拟数据: 获取到 5 条评论")


def demo_headers():
    print("\n" + "=" * 50)
    print("4. 请求头设置")
    print("=" * 50)

    headers = {
        "Accept": "application/json",
        "User-Agent": "PythonDataAnalysis/1.0",
        "X-Custom-Header": "demo-value",
    }

    resp = fetch_with_fallback(f"{BASE_URL}/posts/1", headers=headers)
    if resp is not None:
        print(f"请求成功，状态码: {resp.status_code}")
        print("响应头:")
        for key in ["Content-Type", "X-Ratelimit-Limit", "X-Powered-By"]:
            val = resp.headers.get(key, "未设置")
            print(f"  {key}: {val}")
    else:
        print("网络不可用，模拟请求头演示")
        print("常见请求头:")
        print("  Accept: application/json")
        print("  Content-Type: application/json")
        print("  Authorization: Bearer <token>")


def demo_content_type():
    print("\n" + "=" * 50)
    print("5. Content-Type与请求体格式")
    print("=" * 50)

    data = {"title": "测试", "body": "内容", "userId": 1}

    print("JSON格式 (application/json):")
    import json
    json_body = json.dumps(data, ensure_ascii=False)
    print(f"  请求体: {json_body}")

    print("\n表单格式 (application/x-www-form-urlencoded):")
    form_body = urlencode(data)
    print(f"  请求体: {form_body}")

    try:
        resp_json = requests.post(
            f"{BASE_URL}/posts", json=data, timeout=5
        )
        print(f"\nJSON请求 Content-Type: {resp_json.request.headers.get('Content-Type')}")

        resp_form = requests.post(
            f"{BASE_URL}/posts", data=data, timeout=5
        )
        print(f"表单请求 Content-Type: {resp_form.request.headers.get('Content-Type')}")
    except (requests.RequestException, Exception):
        print("\n网络不可用，模拟演示:")
        print("  json=参数 → Content-Type: application/json")
        print("  data=参数 → Content-Type: application/x-www-form-urlencoded")


if __name__ == "__main__":
    demo_query_params()
    demo_url_construction()
    demo_path_params()
    demo_headers()
    demo_content_type()
