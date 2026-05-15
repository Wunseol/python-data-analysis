# 数据来源: JSONPlaceholder (https://jsonplaceholder.typicode.com) + 模拟数据

import json
import csv
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BASE_URL = "https://jsonplaceholder.typicode.com"

MOCK_USERS = [
    {"id": 1, "name": "张三", "email": "zhangsan@example.com", "company": {"name": "公司A"}},
    {"id": 2, "name": "李四", "email": "lisi@example.com", "company": {"name": "公司B"}},
    {"id": 3, "name": "王五", "email": "wangwu@example.com", "company": {"name": "公司C"}},
]

MOCK_POSTS = [
    {"userId": 1, "id": 1, "title": "数据分析入门", "body": "内容1"},
    {"userId": 1, "id": 2, "title": "Python实战", "body": "内容2"},
    {"userId": 2, "id": 3, "title": "API开发指南", "body": "内容3"},
    {"userId": 3, "id": 4, "title": "机器学习基础", "body": "内容4"},
]

OUTPUT_DIR = Path(__file__).parent / "output"


def fetch_data(endpoint):
    if HAS_REQUESTS:
        try:
            resp = requests.get(f"{BASE_URL}/{endpoint}", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, Exception):
            pass
    return None


def demo_json_to_dataframe():
    print("=" * 50)
    print("1. API响应转DataFrame")
    print("=" * 50)

    posts = fetch_data("posts?_limit=5")
    if posts is None:
        posts = MOCK_POSTS
        print("使用模拟数据")

    if HAS_PANDAS:
        df = pd.DataFrame(posts)
        print(f"DataFrame形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"\n前3行:")
        print(df.head(3).to_string())
        print(f"\n数据类型:\n{df.dtypes}")
    else:
        print("pandas未安装，展示原始数据:")
        for p in posts[:3]:
            print(f"  {p}")


def demo_nested_json_flatten():
    print("\n" + "=" * 50)
    print("2. 嵌套JSON扁平化处理")
    print("=" * 50)

    users = fetch_data("users?_limit=3")
    if users is None:
        users = MOCK_USERS
        print("使用模拟数据")

    flat_users = []
    for u in users:
        flat = {
            "id": u["id"],
            "name": u["name"],
            "email": u["email"],
            "company_name": u.get("company", {}).get("name", ""),
        }
        flat_users.append(flat)

    if HAS_PANDAS:
        df = pd.DataFrame(flat_users)
        print("扁平化后的DataFrame:")
        print(df.to_string())
    else:
        print("扁平化后的数据:")
        for u in flat_users:
            print(f"  {u}")


def demo_save_json():
    print("\n" + "=" * 50)
    print("3. 保存为JSON文件")
    print("=" * 50)

    posts = fetch_data("posts?_limit=5")
    if posts is None:
        posts = MOCK_POSTS
        print("使用模拟数据")

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "posts.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_file}")

    compact_file = OUTPUT_DIR / "posts_compact.json"
    with open(compact_file, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, separators=(",", ":"))
    print(f"紧凑格式: {compact_file}")

    import os
    print(f"格式化大小: {os.path.getsize(output_file)} 字节")
    print(f"紧凑大小: {os.path.getsize(compact_file)} 字节")


def demo_save_csv():
    print("\n" + "=" * 50)
    print("4. 保存为CSV文件")
    print("=" * 50)

    posts = fetch_data("posts?_limit=5")
    if posts is None:
        posts = MOCK_POSTS
        print("使用模拟数据")

    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_file = OUTPUT_DIR / "posts.csv"
    if posts:
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)
        print(f"已保存: {csv_file}")

        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"验证: 读取到 {len(rows)} 行")

    if HAS_PANDAS:
        df = pd.DataFrame(posts)
        csv_file2 = OUTPUT_DIR / "posts_pandas.csv"
        df.to_csv(csv_file2, index=False, encoding="utf-8-sig")
        print(f"pandas保存: {csv_file2}")


def demo_format_conversion():
    print("\n" + "=" * 50)
    print("5. 数据格式互转")
    print("=" * 50)

    posts = fetch_data("posts?_limit=3")
    if posts is None:
        posts = MOCK_POSTS[:3]
        print("使用模拟数据")

    OUTPUT_DIR.mkdir(exist_ok=True)

    if HAS_PANDAS:
        df = pd.DataFrame(posts)

        df.to_json(OUTPUT_DIR / "from_df.json", orient="records", force_ascii=False)
        print("DataFrame → JSON (records格式)")

        df.to_csv(OUTPUT_DIR / "from_df.csv", index=False, encoding="utf-8-sig")
        print("DataFrame → CSV")

        df.to_dict(orient="records")
        print("DataFrame → dict")

        json_str = df.to_json(orient="records", force_ascii=False)
        df_back = pd.read_json(json_str)
        print(f"JSON → DataFrame: {df_back.shape}")

        print("\norient格式对比:")
        for orient in ["records", "index", "columns", "values"]:
            result = df.to_json(orient=orient, force_ascii=False)
            print(f"  {orient}: {result[:60]}...")
    else:
        print("pandas未安装，基础格式转换:")
        print(f"  list长度: {len(posts)}")
        print(f"  第一条: {posts[0]}")


def demo_data_cleaning():
    print("\n" + "=" * 50)
    print("6. API数据清洗与格式化")
    print("=" * 50)

    posts = fetch_data("posts?_limit=5")
    if posts is None:
        posts = MOCK_POSTS
        print("使用模拟数据")

    if HAS_PANDAS:
        df = pd.DataFrame(posts)

        print("原始数据:")
        print(df[["id", "title"]].to_string())

        df["title_length"] = df["title"].str.len()
        df["body_length"] = df["body"].str.len()
        df["title_preview"] = df["title"].str[:20]

        print("\n添加计算列后:")
        print(df[["id", "title_preview", "title_length", "body_length"]].to_string())

        print(f"\n统计信息:")
        print(f"  标题平均长度: {df['title_length'].mean():.1f}")
        print(f"  正文平均长度: {df['body_length'].mean():.1f}")
    else:
        print("数据清洗结果:")
        for p in posts:
            print(f"  ID={p['id']}, 标题长度={len(p['title'])}, 正文长度={len(p['body'])}")


if __name__ == "__main__":
    demo_json_to_dataframe()
    demo_nested_json_flatten()
    demo_save_json()
    demo_save_csv()
    demo_format_conversion()
    demo_data_cleaning()
