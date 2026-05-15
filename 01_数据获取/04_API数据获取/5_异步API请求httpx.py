# 数据来源: JSONPlaceholder (https://jsonplaceholder.typicode.com) + 模拟API

import asyncio
import json
import time
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BASE_URL = "https://jsonplaceholder.typicode.com"

MOCK_POSTS = [
    {"userId": i, "id": i, "title": f"模拟文章{i}", "body": f"内容{i}"}
    for i in range(1, 11)
]


def demo_sync_requests():
    print("=" * 50)
    print("1. 同步请求对比（requests）")
    print("=" * 50)

    if not HAS_REQUESTS:
        print("requests未安装，使用模拟数据演示")
        start = time.time()
        results = []
        for i in range(1, 6):
            results.append(MOCK_POSTS[i - 1])
        elapsed = time.time() - start
        print(f"模拟5次串行请求，耗时: {elapsed:.3f}秒")
        for r in results:
            print(f"  ID={r['id']}: {r['title']}")
        return

    start = time.time()
    results = []
    for post_id in range(1, 6):
        try:
            resp = requests.get(f"{BASE_URL}/posts/{post_id}", timeout=5)
            resp.raise_for_status()
            results.append(resp.json())
        except (requests.RequestException, Exception):
            results.append(MOCK_POSTS[post_id - 1])
    elapsed = time.time() - start

    print(f"5次串行请求，耗时: {elapsed:.3f}秒")
    for r in results:
        print(f"  ID={r['id']}: {r['title'][:30]}")


async def demo_async_httpx():
    print("\n" + "=" * 50)
    print("2. 异步请求（httpx async/await）")
    print("=" * 50)

    if not HAS_HTTPX:
        print("httpx未安装，使用模拟数据演示异步概念")
        print("\n异步请求核心模式:")
        print("  async with httpx.AsyncClient() as client:")
        print("      response = await client.get(url)")
        print("\n模拟5次并发请求:")
        start = time.time()
        await asyncio.sleep(0.01)
        results = MOCK_POSTS[:5]
        elapsed = time.time() - start
        print(f"模拟并发请求，耗时: {elapsed:.3f}秒")
        for r in results:
            print(f"  ID={r['id']}: {r['title']}")
        return

    async with httpx.AsyncClient(timeout=10) as client:
        start = time.time()
        tasks = []
        for post_id in range(1, 6):
            task = client.get(f"{BASE_URL}/posts/{post_id}")
            tasks.append(task)

        try:
            responses = await asyncio.gather(*tasks)
            results = [r.json() for r in responses if r.status_code == 200]
        except (httpx.HTTPError, Exception):
            results = MOCK_POSTS[:5]

        elapsed = time.time() - start
        print(f"5次并发请求，耗时: {elapsed:.3f}秒")
        for r in results:
            print(f"  ID={r['id']}: {r['title'][:30]}")


async def demo_concurrent_requests():
    print("\n" + "=" * 50)
    print("3. 批量并发请求")
    print("=" * 50)

    urls = [f"{BASE_URL}/posts/{i}" for i in range(1, 11)]
    print(f"准备请求 {len(urls)} 个URL")

    if not HAS_HTTPX:
        print("httpx未安装，模拟并发演示")
        start = time.time()
        await asyncio.sleep(0.02)
        results = MOCK_POSTS[:10]
        elapsed = time.time() - start
        print(f"模拟10次并发请求，耗时: {elapsed:.3f}秒")
        print(f"获取 {len(results)} 条数据")
        return

    async with httpx.AsyncClient(timeout=10) as client:
        start = time.time()

        semaphore = asyncio.Semaphore(5)

        async def limited_fetch(url):
            async with semaphore:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.HTTPError, Exception):
                    post_id = int(url.split("/")[-1])
                    return MOCK_POSTS[post_id - 1]

        tasks = [limited_fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"10次并发请求（限制5并发），耗时: {elapsed:.3f}秒")
        print(f"获取 {len(results)} 条数据")


async def demo_async_with_retry():
    print("\n" + "=" * 50)
    print("4. 异步请求与重试机制")
    print("=" * 50)

    max_retries = 3
    retry_delay = 0.5

    if not HAS_HTTPX:
        print("httpx未安装，模拟重试逻辑演示")
        for attempt in range(1, max_retries + 1):
            print(f"  第{attempt}次尝试...")
            if attempt < 3:
                print(f"  请求失败，{retry_delay}秒后重试")
                await asyncio.sleep(0.01)
            else:
                print("  ✓ 请求成功")
                break
        return

    async with httpx.AsyncClient(timeout=10) as client:
        url = f"{BASE_URL}/posts/1"
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                print(f"第{attempt}次尝试成功: {resp.json()['title'][:30]}")
                break
            except (httpx.HTTPError, Exception) as e:
                print(f"第{attempt}次尝试失败: {e}")
                if attempt < max_retries:
                    print(f"  {retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)
                else:
                    print("  达到最大重试次数，使用模拟数据")
                    print(f"  模拟数据: {MOCK_POSTS[0]['title']}")


async def demo_async_save():
    print("\n" + "=" * 50)
    print("5. 异步获取并保存数据")
    print("=" * 50)

    if not HAS_HTTPX:
        results = MOCK_POSTS[:5]
        print(f"模拟获取 {len(results)} 条数据")
    else:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                tasks = [client.get(f"{BASE_URL}/posts/{i}") for i in range(1, 6)]
                responses = await asyncio.gather(*tasks)
                results = [r.json() for r in responses if r.status_code == 200]
            except (httpx.HTTPError, Exception):
                results = MOCK_POSTS[:5]
            print(f"获取 {len(results)} 条数据")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "async_posts.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_file}")


async def main():
    demo_sync_requests()
    await demo_async_httpx()
    await demo_concurrent_requests()
    await demo_async_with_retry()
    await demo_async_save()


if __name__ == "__main__":
    asyncio.run(main())
