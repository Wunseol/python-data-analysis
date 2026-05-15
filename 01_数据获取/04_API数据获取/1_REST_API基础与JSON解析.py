# 数据来源: JSONPlaceholder (https://jsonplaceholder.typicode.com)
# 依赖库最低版本要求: requests>=2.28, httpx>=0.24

import json
import requests
from pathlib import Path

BASE_URL = "https://jsonplaceholder.typicode.com"

MOCK_POSTS = [
    {"userId": 1, "id": 1, "title": "模拟文章标题1", "body": "这是模拟的文章内容1"},
    {"userId": 1, "id": 2, "title": "模拟文章标题2", "body": "这是模拟的文章内容2"},
    {"userId": 2, "id": 3, "title": "模拟文章标题3", "body": "这是模拟的文章内容3"},
]

MOCK_CREATED = {"userId": 1, "id": 101, "title": "新文章", "body": "新文章内容"}


def fetch_with_fallback(url, timeout=5):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    except (requests.RequestException, Exception):
        return None


def demo_get_request():
    print("=" * 50)
    print("1. GET请求 - 获取文章列表")
    print("=" * 50)

    resp = fetch_with_fallback(f"{BASE_URL}/posts")
    if resp is not None:
        print(f"状态码: {resp.status_code}")
        print(f"响应类型: {resp.headers.get('Content-Type')}")
        posts = resp.json()
        print(f"获取到 {len(posts)} 篇文章")
        print(f"第一篇: {posts[0]['title']}")
    else:
        print("网络不可用，使用模拟数据")
        print(f"获取到 {len(MOCK_POSTS)} 篇文章")
        print(f"第一篇: {MOCK_POSTS[0]['title']}")


def demo_json_parsing():
    print("\n" + "=" * 50)
    print("2. JSON解析 - 多种解析方式")
    print("=" * 50)

    resp = fetch_with_fallback(f"{BASE_URL}/posts/1")
    if resp is not None:
        data_json = resp.json()
        data_text = json.loads(resp.text)
        print(f"resp.json(): {data_json['title']}")
        print(f"json.loads(): {data_text['title']}")
        print(f"两者结果一致: {data_json == data_text}")

        raw_text = resp.text
        print(f"原始响应长度: {len(raw_text)} 字符")
        print(f"原始响应前80字符: {raw_text[:80]}...")
    else:
        mock_json = json.dumps(MOCK_POSTS[0], ensure_ascii=False)
        data = json.loads(mock_json)
        print(f"模拟数据解析: {data['title']}")


def demo_status_codes():
    print("\n" + "=" * 50)
    print("3. 状态码处理")
    print("=" * 50)

    status_descriptions = {
        200: "请求成功",
        201: "资源创建成功",
        400: "请求参数错误",
        401: "未授权",
        403: "禁止访问",
        404: "资源不存在",
        500: "服务器内部错误",
    }

    for code, desc in status_descriptions.items():
        print(f"  {code}: {desc}")

    resp = fetch_with_fallback(f"{BASE_URL}/posts/1")
    if resp is not None:
        if resp.status_code == 200:
            print(f"\n实际请求状态码: {resp.status_code} ✓")
        elif resp.status_code == 404:
            print(f"\n资源未找到: {resp.status_code}")
        else:
            print(f"\n其他状态码: {resp.status_code}")
    else:
        print("\n网络不可用，模拟状态码: 200 ✓")


def demo_post_request():
    print("\n" + "=" * 50)
    print("4. POST请求 - 创建资源")
    print("=" * 50)

    new_post = {"title": "新文章", "body": "文章内容", "userId": 1}

    try:
        resp = requests.post(
            f"{BASE_URL}/posts",
            json=new_post,
            timeout=5,
        )
        resp.raise_for_status()
        print(f"状态码: {resp.status_code}")
        created = resp.json()
        print(f"创建成功，ID: {created['id']}")
        print(f"返回数据: {json.dumps(created, ensure_ascii=False, indent=2)}")
    except (requests.RequestException, Exception):
        print("网络不可用，使用模拟数据")
        print(f"模拟创建成功，ID: {MOCK_CREATED['id']}")
        print(f"返回数据: {json.dumps(MOCK_CREATED, ensure_ascii=False, indent=2)}")


def demo_save_json():
    print("\n" + "=" * 50)
    print("5. 保存JSON到文件")
    print("=" * 50)

    resp = fetch_with_fallback(f"{BASE_URL}/posts/1")
    if resp is not None:
        data = resp.json()
    else:
        data = MOCK_POSTS[0]

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "post_1.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_file}")

    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"验证读取: {loaded['title']}")


if __name__ == "__main__":
    demo_get_request()
    demo_json_parsing()
    demo_status_codes()
    demo_post_request()
    demo_save_json()
