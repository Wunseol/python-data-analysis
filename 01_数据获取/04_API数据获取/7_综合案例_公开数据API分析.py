# 数据来源: JSONPlaceholder (https://jsonplaceholder.typicode.com), Open-Meteo (https://api.open-meteo.com)

import json
import time
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BASE_URL_JSONPLACEHOLDER = "https://jsonplaceholder.typicode.com"
BASE_URL_OPEN_METEO = "https://api.open-meteo.com/v1"

OUTPUT_DIR = Path(__file__).parent / "output"

MOCK_POSTS = [
    {"userId": 1, "id": i, "title": f"文章标题{i}", "body": f"文章内容{i}" * 5}
    for i in range(1, 11)
] + [
    {"userId": 2, "id": i, "title": f"技术分享{i}", "body": f"技术内容{i}" * 8}
    for i in range(11, 21)
] + [
    {"userId": 3, "id": i, "title": f"学习笔记{i}", "body": f"笔记内容{i}" * 3}
    for i in range(21, 31)
]

MOCK_COMMENTS = [
    {"postId": i, "id": i * 5 + j, "name": f"评论{j}", "email": f"user{j}@example.com", "body": f"评论内容{i}-{j}"}
    for i in range(1, 6) for j in range(1, 6)
]

MOCK_WEATHER = {
    "latitude": 39.9,
    "longitude": 116.4,
    "daily": {
        "time": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05",
                 "2025-01-06", "2025-01-07"],
        "temperature_2m_max": [2.1, 4.3, 1.5, -0.8, 3.2, 5.1, 2.8],
        "temperature_2m_min": [-6.2, -4.1, -7.3, -9.5, -5.0, -3.2, -5.8],
        "precipitation_sum": [0.0, 2.1, 0.5, 0.0, 1.2, 0.0, 3.5],
    },
}


def fetch_with_fallback(url, params=None, timeout=8):
    if not HAS_REQUESTS:
        return None
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, Exception):
        return None


def step1_fetch_posts():
    print("=" * 60)
    print("步骤1: 获取文章数据")
    print("=" * 60)

    data = fetch_with_fallback(f"{BASE_URL_JSONPLACEHOLDER}/posts")
    if data is not None:
        print(f"从API获取 {len(data)} 篇文章")
        return data
    else:
        print(f"使用模拟数据: {len(MOCK_POSTS)} 篇文章")
        return MOCK_POSTS


def step2_fetch_comments():
    print("\n" + "=" * 60)
    print("步骤2: 获取评论数据")
    print("=" * 60)

    data = fetch_with_fallback(f"{BASE_URL_JSONPLACEHOLDER}/comments")
    if data is not None:
        print(f"从API获取 {len(data)} 条评论")
        return data
    else:
        print(f"使用模拟数据: {len(MOCK_COMMENTS)} 条评论")
        return MOCK_COMMENTS


def step3_fetch_weather():
    print("\n" + "=" * 60)
    print("步骤3: 获取天气数据（北京近7天）")
    print("=" * 60)

    params = {
        "latitude": 39.9042,
        "longitude": 116.4074,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Shanghai",
        "forecast_days": 7,
    }

    data = fetch_with_fallback(f"{BASE_URL_OPEN_METEO}/forecast", params=params)
    if data is not None:
        print(f"从Open-Meteo获取天气数据")
        return data
    else:
        print("使用模拟天气数据")
        return MOCK_WEATHER


def step4_analyze_posts(posts):
    print("\n" + "=" * 60)
    print("步骤4: 文章数据分析")
    print("=" * 60)

    if HAS_PANDAS:
        df = pd.DataFrame(posts)
        print(f"总文章数: {len(df)}")

        user_counts = df["userId"].value_counts().sort_index()
        print(f"\n各用户文章数:")
        for uid, count in user_counts.items():
            print(f"  用户{uid}: {count}篇")

        df["title_len"] = df["title"].str.len()
        df["body_len"] = df["body"].str.len()

        print(f"\n标题长度统计:")
        print(f"  平均: {df['title_len'].mean():.1f}")
        print(f"  最大: {df['title_len'].max()}")
        print(f"  最小: {df['title_len'].min()}")

        print(f"\n正文长度统计:")
        print(f"  平均: {df['body_len'].mean():.1f}")
        print(f"  最大: {df['body_len'].max()}")
        print(f"  最小: {df['body_len'].min()}")

        return df
    else:
        print(f"总文章数: {len(posts)}")
        user_counts = {}
        for p in posts:
            uid = p["userId"]
            user_counts[uid] = user_counts.get(uid, 0) + 1
        print(f"\n各用户文章数:")
        for uid in sorted(user_counts):
            print(f"  用户{uid}: {user_counts[uid]}篇")
        return posts


def step5_analyze_comments(comments):
    print("\n" + "=" * 60)
    print("步骤5: 评论数据分析")
    print("=" * 60)

    if HAS_PANDAS:
        df = pd.DataFrame(comments)
        print(f"总评论数: {len(df)}")

        post_comment_counts = df["postId"].value_counts().sort_index().head(5)
        print(f"\n各文章评论数（前5篇）:")
        for pid, count in post_comment_counts.items():
            print(f"  文章{pid}: {count}条")

        email_domains = df["email"].str.split("@").str[1].value_counts().head(5)
        print(f"\n邮箱域名分布:")
        for domain, count in email_domains.items():
            print(f"  {domain}: {count}")

        return df
    else:
        print(f"总评论数: {len(comments)}")
        post_counts = {}
        for c in comments:
            pid = c["postId"]
            post_counts[pid] = post_counts.get(pid, 0) + 1
        print(f"\n各文章评论数:")
        for pid in sorted(post_counts)[:5]:
            print(f"  文章{pid}: {post_counts[pid]}条")
        return comments


def step6_analyze_weather(weather):
    print("\n" + "=" * 60)
    print("步骤6: 天气数据分析")
    print("=" * 60)

    daily = weather.get("daily", {})
    dates = daily.get("time", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])

    if HAS_PANDAS:
        df = pd.DataFrame({
            "日期": dates,
            "最高温": temp_max,
            "最低温": temp_min,
            "降水量": precip,
        })

        print("天气数据:")
        print(df.to_string(index=False))

        df["温差"] = df["最高温"] - df["最低温"]
        print(f"\n统计摘要:")
        print(f"  平均最高温: {df['最高温'].mean():.1f}°C")
        print(f"  平均最低温: {df['最低温'].mean():.1f}°C")
        print(f"  最大温差: {df['温差'].max():.1f}°C")
        print(f"  总降水量: {df['降水量'].sum():.1f}mm")
        print(f"  降水天数: {(df['降水量'] > 0).sum()}天")

        return df
    else:
        print("日期       最高温  最低温  降水量")
        for i in range(len(dates)):
            print(f"{dates[i]}  {temp_max[i]:6.1f}  {temp_min[i]:6.1f}  {precip[i]:6.1f}")

        avg_max = sum(temp_max) / len(temp_max)
        avg_min = sum(temp_min) / len(temp_min)
        total_precip = sum(precip)
        print(f"\n平均最高温: {avg_max:.1f}°C, 平均最低温: {avg_min:.1f}°C")
        print(f"总降水量: {total_precip:.1f}mm")
        return None


def step7_visualize(posts_df, comments_df, weather_df):
    print("\n" + "=" * 60)
    print("步骤7: 数据可视化")
    print("=" * 60)

    if not HAS_MATPLOTLIB or not HAS_PANDAS:
        print("matplotlib或pandas未安装，跳过可视化")
        print("安装命令: pip install matplotlib pandas")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("公开数据API综合分析", fontsize=16)

    if isinstance(posts_df, pd.DataFrame):
        user_counts = posts_df["userId"].value_counts().sort_index()
        axes[0, 0].bar(user_counts.index.astype(str), user_counts.values, color="steelblue")
        axes[0, 0].set_title("各用户文章数")
        axes[0, 0].set_xlabel("用户ID")
        axes[0, 0].set_ylabel("文章数")

    if isinstance(comments_df, pd.DataFrame):
        top_posts = comments_df["postId"].value_counts().head(5)
        axes[0, 1].barh(top_posts.index.astype(str), top_posts.values, color="coral")
        axes[0, 1].set_title("评论最多的文章（Top5）")
        axes[0, 1].set_xlabel("评论数")
        axes[0, 1].set_ylabel("文章ID")

    if isinstance(posts_df, pd.DataFrame):
        axes[1, 0].hist(posts_df["title_len"], bins=10, color="seagreen", edgecolor="white")
        axes[1, 0].set_title("文章标题长度分布")
        axes[1, 0].set_xlabel("标题长度")
        axes[1, 0].set_ylabel("频次")

    if isinstance(weather_df, pd.DataFrame):
        x = range(len(weather_df))
        axes[1, 1].plot(x, weather_df["最高温"], "r-o", label="最高温", markersize=5)
        axes[1, 1].plot(x, weather_df["最低温"], "b-o", label="最低温", markersize=5)
        axes[1, 1].fill_between(x, weather_df["最低温"], weather_df["最高温"], alpha=0.2)
        axes[1, 1].set_title("北京近7天气温变化")
        axes[1, 1].set_xlabel("日期")
        axes[1, 1].set_ylabel("温度(°C)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(weather_df["日期"], rotation=45, fontsize=8)
        axes[1, 1].legend()

    plt.tight_layout()
    chart_file = OUTPUT_DIR / "api_analysis.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {chart_file}")


def step8_save_all(posts, comments, weather):
    print("\n" + "=" * 60)
    print("步骤8: 保存所有数据")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_data = {
        "posts": posts,
        "comments": comments,
        "weather": weather,
        "metadata": {
            "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": ["JSONPlaceholder", "Open-Meteo"],
        },
    }

    output_file = OUTPUT_DIR / "comprehensive_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"综合数据已保存: {output_file}")

    if HAS_PANDAS:
        posts_df = pd.DataFrame(posts)
        posts_df.to_csv(OUTPUT_DIR / "analysis_posts.csv", index=False, encoding="utf-8-sig")
        print(f"文章CSV已保存: {OUTPUT_DIR / 'analysis_posts.csv'}")

        comments_df = pd.DataFrame(comments)
        comments_df.to_csv(OUTPUT_DIR / "analysis_comments.csv", index=False, encoding="utf-8-sig")
        print(f"评论CSV已保存: {OUTPUT_DIR / 'analysis_comments.csv'}")


def main():
    print("综合案例: 公开数据API分析")
    print("数据源: JSONPlaceholder + Open-Meteo")
    print()

    posts = step1_fetch_posts()
    comments = step2_fetch_comments()
    weather = step3_fetch_weather()

    posts_df = step4_analyze_posts(posts)
    comments_df = step5_analyze_comments(comments)
    weather_df = step6_analyze_weather(weather)

    step7_visualize(posts_df, comments_df, weather_df)
    step8_save_all(posts, comments, weather)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
