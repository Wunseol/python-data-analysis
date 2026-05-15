# -*- coding: utf-8 -*-
"""
情感分析基础

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import pandas as pd


def demo_textblob():
    print("=" * 60)
    print("1. TextBlob 英文情感分析")
    print("=" * 60)

    try:
        from textblob import TextBlob
    except ImportError:
        print("  TextBlob 未安装，请运行: pip install textblob")
        print("  以下为概念演示:")
        print("""
    from textblob import TextBlob

    blob = TextBlob("I love this product! It's amazing.")
    sentiment = blob.sentiment
    print(f"极性: {sentiment.polarity}")      # -1 到 1，正值=积极
    print(f"主观性: {sentiment.subjectivity}")  # 0 到 1，0=客观，1=主观
        """)
        return

    texts = [
        "I love this product! It's amazing and wonderful.",
        "This is terrible. I hate it so much.",
        "The weather is okay today, nothing special.",
        "The movie was absolutely fantastic! Best film ever!",
        "I'm disappointed with the service quality.",
        "The report provides an objective analysis of the situation.",
    ]

    print(f"  {'文本':<55} {'极性':>6} {'主观性':>6} {'情感'}")
    print("  " + "-" * 80)

    for text in texts:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        if sentiment.polarity > 0.1:
            label = "积极"
        elif sentiment.polarity < -0.1:
            label = "消极"
        else:
            label = "中性"
        print(f"  {text:<55} {sentiment.polarity:>6.3f} {sentiment.subjectivity:>6.3f} {label}")

    print("\n  极性 (polarity): 范围 [-1, 1]，正值表示积极，负值表示消极")
    print("  主观性 (subjectivity): 范围 [0, 1]，0=客观事实，1=完全主观")


def demo_snownlp():
    print("\n" + "=" * 60)
    print("2. SnowNLP 中文情感分析")
    print("=" * 60)

    try:
        from snownlp import SnowNLP
    except ImportError:
        print("  SnowNLP 未安装，请运行: pip install snownlp")
        print("  以下为概念演示:")
        print("""
    from snownlp import SnowNLP

    s = SnowNLP("这部电影真的很好看，强烈推荐！")
    print(f"情感分数: {s.sentiments}")  # 0 到 1，越接近1越积极
        """)
        return

    texts = [
        "这部电影真的很好看，强烈推荐！",
        "服务态度太差了，再也不来了。",
        "今天天气一般般，没什么特别的感觉。",
        "产品质量非常好，物超所值！",
        "快递太慢了，等了好久才收到，很失望。",
        "这本书内容丰富，值得一读。",
        "餐厅环境优雅，菜品味道不错。",
        "这次体验非常糟糕，完全不值这个价格。",
    ]

    print(f"  {'文本':<35} {'情感分数':>8} {'情感倾向'}")
    print("  " + "-" * 60)

    for text in texts:
        s = SnowNLP(text)
        score = s.sentiments
        if score > 0.6:
            label = "积极 😊"
        elif score < 0.4:
            label = "消极 😞"
        else:
            label = "中性 😐"
        print(f"  {text:<35} {score:>8.4f} {label}")

    print("\n  SnowNLP 情感分数: 范围 [0, 1]")
    print("    接近 1 → 积极情感")
    print("    接近 0 → 消极情感")
    print("    0.5 附近 → 中性情感")


def demo_batch_sentiment():
    print("\n" + "=" * 60)
    print("3. 批量情感分析")
    print("=" * 60)

    try:
        from snownlp import SnowNLP
    except ImportError:
        print("  SnowNLP 未安装，跳过批量情感分析演示")
        return

    reviews = [
        "这个手机用起来很流畅，拍照效果也很好",
        "电池续航太差了，一天要充两次电",
        "性价比不错，推荐购买",
        "屏幕容易碎，质量堪忧",
        "系统更新后变得很卡顿",
        "外观设计很漂亮，手感也很好",
        "客服态度很好，问题解决很快",
        "物流太慢了，包装也有破损",
        "功能齐全，操作简单易懂",
        "价格偏贵，不太值得",
        "音质清晰，通话效果不错",
        "散热效果不好，玩游戏容易发烫",
    ]

    results = []
    for review in reviews:
        s = SnowNLP(review)
        score = s.sentiments
        if score > 0.6:
            sentiment = "积极"
        elif score < 0.4:
            sentiment = "消极"
        else:
            sentiment = "中性"
        results.append({
            "评论": review,
            "情感分数": round(score, 4),
            "情感倾向": sentiment,
        })

    df = pd.DataFrame(results)
    print(f"  批量情感分析结果:")
    print(df.to_string(index=False))

    print(f"\n  统计摘要:")
    sentiment_counts = df["情感倾向"].value_counts()
    for s, count in sentiment_counts.items():
        print(f"    {s}: {count}条 ({count / len(df):.1%})")
    print(f"    平均情感分数: {df['情感分数'].mean():.4f}")
    print(f"    情感分数标准差: {df['情感分数'].std():.4f}")


def demo_sentiment_visualization():
    print("\n" + "=" * 60)
    print("4. 情感分析结果可视化")
    print("=" * 60)

    try:
        from snownlp import SnowNLP
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  所需库未安装，跳过可视化演示")
        return

    reviews = [
        "这个手机用起来很流畅，拍照效果也很好",
        "电池续航太差了，一天要充两次电",
        "性价比不错，推荐购买",
        "屏幕容易碎，质量堪忧",
        "系统更新后变得很卡顿",
        "外观设计很漂亮，手感也很好",
        "客服态度很好，问题解决很快",
        "物流太慢了，包装也有破损",
        "功能齐全，操作简单易懂",
        "价格偏贵，不太值得",
    ]

    scores = [SnowNLP(r).sentiments for r in reviews]

    import tempfile
    tmp_dir = tempfile.mkdtemp()
    import os

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(scores, bins=10, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="中性线")
    axes[0].set_xlabel("情感分数")
    axes[0].set_ylabel("数量")
    axes[0].set_title("情感分数分布")
    axes[0].legend()

    colors = ["#ff6b6b" if s < 0.4 else "#ffd93d" if s < 0.6 else "#6bcb77" for s in scores]
    axes[1].barh(range(len(reviews)), scores, color=colors)
    axes[1].set_yticks(range(len(reviews)))
    axes[1].set_yticklabels([r[:10] + "..." for r in reviews], fontsize=8)
    axes[1].axvline(x=0.5, color="red", linestyle="--")
    axes[1].set_xlabel("情感分数")
    axes[1].set_title("各评论情感分数")

    plt.tight_layout()
    fig_path = os.path.join(tmp_dir, "sentiment_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  情感分析可视化图已保存: {fig_path}")


def demo_sentiment_comparison():
    print("\n" + "=" * 60)
    print("5. TextBlob vs SnowNLP 对比")
    print("=" * 60)

    print("  特性对比:")
    comparison = {
        "支持语言": ("英文", "中文"),
        "极性范围": ("[-1, 1]", "[0, 1]"),
        "主观性": ("支持", "不支持"),
        "分词功能": ("内置", "内置"),
        "关键词提取": ("不支持", "支持"),
        "摘要功能": ("不支持", "支持"),
        "训练数据": ("电影评论", "商品评论"),
        "可自定义训练": ("较复杂", "较简单"),
    }
    print(f"  {'特性':<15} {'TextBlob':<15} {'SnowNLP':<15}")
    print("  " + "-" * 45)
    for feature, (tb, sn) in comparison.items():
        print(f"  {feature:<15} {tb:<15} {sn:<15}")

    print("\n  选型建议:")
    print("    - 英文文本 → TextBlob")
    print("    - 中文文本 → SnowNLP")
    print("    - 需要更高精度 → 考虑基于深度学习的方案 (如 BERT 微调)")
    print("    - 大规模生产环境 → 考虑商用情感分析 API")


if __name__ == "__main__":
    demo_textblob()
    demo_snownlp()
    demo_batch_sentiment()
    demo_sentiment_visualization()
    demo_sentiment_comparison()

    print("\n" + "=" * 60)
    print("情感分析基础 演示完成")
    print("=" * 60)
