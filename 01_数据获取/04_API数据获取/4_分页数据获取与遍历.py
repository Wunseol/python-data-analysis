# 数据来源: JSONPlaceholder (https://jsonplaceholder.typicode.com) + 模拟分页API

import requests
import json
from pathlib import Path

BASE_URL = "https://jsonplaceholder.typicode.com"

MOCK_ITEMS = [
    {"id": i, "name": f"项目{i}", "value": i * 10, "category": f"类别{i % 3}"}
    for i in range(1, 26)
]


def fetch_with_fallback(url, params=None, timeout=5):
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except (requests.RequestException, Exception):
        return None


def demo_offset_limit_pagination():
    print("=" * 50)
    print("1. Offset/Limit分页模式")
    print("=" * 50)

    page_size = 5
    all_posts = []
    page = 0

    while True:
        offset = page * page_size
        params = {"_start": offset, "_limit": page_size}

        resp = fetch_with_fallback(f"{BASE_URL}/posts", params=params)
        if resp is not None:
            posts = resp.json()
            if not posts:
                break
            all_posts.extend(posts)
            print(f"第{page + 1}页: 获取 {len(posts)} 条 (offset={offset})")
            if len(posts) < page_size:
                break
        else:
            start = offset
            end = min(offset + page_size, len(MOCK_ITEMS))
            page_items = MOCK_ITEMS[start:end]
            if not page_items:
                break
            all_posts.extend(page_items)
            print(f"第{page + 1}页: 获取 {len(page_items)} 条 (offset={offset}) [模拟]")
            if len(page_items) < page_size:
                break

        page += 1
        if page >= 5:
            break

    print(f"\n总计获取: {len(all_posts)} 条数据")


def demo_cursor_pagination():
    print("\n" + "=" * 50)
    print("2. Cursor分页模式（模拟）")
    print("=" * 50)

    class MockCursorAPI:
        def __init__(self, items, page_size=5):
            self.items = items
            self.page_size = page_size

        def get_page(self, cursor=0):
            start = cursor
            end = min(start + self.page_size, len(self.items))
            page_data = self.items[start:end]
            next_cursor = end if end < len(self.items) else None
            return {
                "data": page_data,
                "next_cursor": next_cursor,
                "has_more": next_cursor is not None,
            }

    api = MockCursorAPI(MOCK_ITEMS, page_size=5)
    all_items = []
    cursor = 0
    page_num = 0

    while True:
        result = api.get_page(cursor)
        items = result["data"]
        all_items.extend(items)
        page_num += 1
        print(f"第{page_num}页: 获取 {len(items)} 条, cursor={cursor}")

        if not result["has_more"]:
            print("已到达最后一页")
            break
        cursor = result["next_cursor"]

    print(f"\n总计获取: {len(all_items)} 条数据")


def demo_page_number_pagination():
    print("\n" + "=" * 50)
    print("3. 页码分页模式（模拟）")
    print("=" * 50)

    class MockPageAPI:
        def __init__(self, items, page_size=5):
            self.items = items
            self.page_size = page_size
            self.total = len(items)
            self.total_pages = (self.total + page_size - 1) // page_size

        def get_page(self, page=1):
            start = (page - 1) * self.page_size
            end = min(start + self.page_size, self.total)
            return {
                "data": self.items[start:end],
                "page": page,
                "per_page": self.page_size,
                "total": self.total,
                "total_pages": self.total_pages,
            }

    api = MockPageAPI(MOCK_ITEMS, page_size=5)
    all_items = []
    page = 1

    while True:
        result = api.get_page(page)
        all_items.extend(result["data"])
        print(
            f"第{result['page']}/{result['total_pages']}页: "
            f"获取 {len(result['data'])} 条, "
            f"总进度 {len(all_items)}/{result['total']}"
        )

        if page >= result["total_pages"]:
            break
        page += 1

    print(f"\n总计获取: {len(all_items)} 条数据")


def demo_accumulate_and_save():
    print("\n" + "=" * 50)
    print("4. 分页数据累积收集与保存")
    print("=" * 50)

    all_users = []
    params = {"_limit": 5}

    resp = fetch_with_fallback(f"{BASE_URL}/users", params=params)
    if resp is not None:
        all_users = resp.json()
        print(f"从API获取 {len(all_users)} 个用户")
    else:
        all_users = [
            {"id": i, "name": f"用户{i}", "email": f"user{i}@example.com"}
            for i in range(1, 6)
        ]
        print(f"模拟获取 {len(all_users)} 个用户")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "paginated_users.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_users, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_file}")

    print("\n用户列表:")
    for u in all_users:
        name = u.get("name", "未知")
        email = u.get("email", "未知")
        print(f"  {u['id']}. {name} ({email})")


if __name__ == "__main__":
    demo_offset_limit_pagination()
    demo_cursor_pagination()
    demo_page_number_pagination()
    demo_accumulate_and_save()
