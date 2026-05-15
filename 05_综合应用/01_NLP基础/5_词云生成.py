# -*- coding: utf-8 -*-
"""
词云生成

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import os
import tempfile
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def demo_basic_wordcloud():
    print("=" * 60)
    print("1. 基本词云生成 (英文)")
    print("=" * 60)

    text = (
        "Python data analysis is becoming increasingly important. "
        "Data science requires Python programming skills. "
        "Machine learning and deep learning are subsets of artificial intelligence. "
        "Natural language processing uses Python extensively. "
        "Data visualization helps understand patterns in data. "
        "Python pandas numpy scikit-learn are essential libraries. "
        "Big data analytics drives business decisions. "
        "Statistical analysis forms the foundation of data science."
    )

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=50,
    )
    wc.generate(text)

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "wordcloud_en.png")
    wc.to_file(output_path)
    print(f"  英文词云已保存: {output_path}")

    print(f"  词云参数:")
    print(f"    尺寸: {wc.width}x{wc.height}")
    print(f"    最大词数: {wc.max_words}")
    print(f"    背景色: {wc.background_color}")

    print("\n  generate_from_text() 方法:")
    wc2 = WordCloud(width=800, height=400, background_color="white")
    wc2.generate_from_text(text)
    print(f"    与 generate() 功能相同，显式指定文本来源")


def demo_chinese_wordcloud():
    print("\n" + "=" * 60)
    print("2. 中文词云生成 (结合 jieba)")
    print("=" * 60)

    text = (
        "自然语言处理是人工智能领域中的重要研究方向。"
        "深度学习技术在自然语言处理中取得了突破性进展。"
        "机器学习算法广泛应用于文本分类和情感分析。"
        "数据分析和数据挖掘是当今最热门的技能之一。"
        "人工智能正在深刻改变我们的生活方式和工作模式。"
        "计算机视觉和自然语言处理是人工智能的两大核心方向。"
        "知识图谱技术为智能问答系统提供了强大的知识支撑。"
        "语音识别技术使得人机交互更加自然和便捷。"
    )

    words = jieba.lcut(text)
    words_filtered = [w for w in words if len(w) > 1]
    text_processed = " ".join(words_filtered)
    print(f"  分词结果: {text_processed[:60]}...")

    font_path = None
    candidate_fonts = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in candidate_fonts:
        if os.path.exists(fp):
            font_path = fp
            print(f"  使用字体: {fp}")
            break

    if font_path is None:
        print("  [警告] 未找到中文字体，中文词云可能无法正常显示")
        print("  请确保系统中有 simhei.ttf 或 msyh.ttc 字体文件")

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        max_words=80,
        colormap="viridis",
    )
    wc.generate(text_processed)

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "wordcloud_cn.png")
    wc.to_file(output_path)
    print(f"  中文词云已保存: {output_path}")


def demo_wordcloud_from_frequencies():
    print("\n" + "=" * 60)
    print("3. 基于词频生成词云")
    print("=" * 60)

    text = (
        "人工智能 深度学习 机器学习 自然语言处理 计算机视觉 "
        "数据挖掘 知识图谱 语音识别 推荐系统 强化学习 "
        "人工智能 人工智能 深度学习 深度学习 机器学习 "
        "自然语言处理 自然语言处理 数据分析 数据分析 数据分析 "
        "深度学习 深度学习 深度学习 人工智能 人工智能 人工智能 "
        "卷积神经网络 循环神经网络 注意力机制 Transformer "
        "BERT GPT 预训练模型 迁移学习 半监督学习 "
    )

    words = text.split()
    word_counts = Counter(words)
    print(f"  词频统计 (Top 10):")
    for word, count in word_counts.most_common(10):
        print(f"    {word}: {count}")

    font_path = None
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(fp):
            font_path = fp
            break

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        max_words=50,
    )
    wc.generate_from_frequencies(word_counts)

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "wordcloud_freq.png")
    wc.to_file(output_path)
    print(f"\n  基于词频的词云已保存: {output_path}")


def demo_wordcloud_mask():
    print("\n" + "=" * 60)
    print("4. 词云蒙版形状")
    print("=" * 60)

    import numpy as np

    mask = np.zeros((400, 600), dtype=np.uint8)
    cx, cy, r = 200, 300, 150
    for i in range(400):
        for j in range(600):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                mask[i, j] = 255

    print(f"  蒙版形状: {mask.shape}")
    print(f"  蒙版类型: {mask.dtype}")
    print(f"  白色区域(绘制区域)像素数: {np.sum(mask == 255)}")

    word_counts = Counter({
        "人工智能": 50, "深度学习": 40, "机器学习": 35,
        "自然语言处理": 30, "计算机视觉": 28, "数据挖掘": 25,
        "知识图谱": 22, "语音识别": 20, "推荐系统": 18,
        "强化学习": 15, "数据分析": 25, "神经网络": 20,
    })

    font_path = None
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(fp):
            font_path = fp
            break

    wc = WordCloud(
        font_path=font_path,
        mask=mask,
        background_color="white",
        max_words=30,
        contour_width=1,
        contour_color="steelblue",
    )
    wc.generate_from_frequencies(word_counts)

    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "wordcloud_mask.png")
    wc.to_file(output_path)
    print(f"  蒙版词云已保存: {output_path}")

    print("\n  使用图片作为蒙版 (概念):")
    print("""
    from PIL import Image
    import numpy as np

    mask_image = np.array(Image.open("shape.png"))
    wc = WordCloud(mask=mask_image, background_color="white")
    wc.generate(text)
    """)


def demo_wordcloud_styles():
    print("\n" + "=" * 60)
    print("5. 词云样式定制")
    print("=" * 60)

    word_counts = Counter({
        "人工智能": 50, "深度学习": 40, "机器学习": 35,
        "自然语言处理": 30, "计算机视觉": 28, "数据挖掘": 25,
        "知识图谱": 22, "语音识别": 20, "推荐系统": 18,
        "强化学习": 15, "数据分析": 25, "神经网络": 20,
        "卷积网络": 18, "循环网络": 16, "注意力机制": 14,
    })

    font_path = None
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(fp):
            font_path = fp
            break

    styles = [
        {
            "name": "暗色主题",
            "params": {
                "background_color": "black",
                "colormap": "Set2",
            },
        },
        {
            "name": "渐变主题",
            "params": {
                "background_color": "white",
                "colormap": "plasma",
            },
        },
        {
            "name": "清新主题",
            "params": {
                "background_color": "#f0f0f0",
                "colormap": "YlGnBu",
            },
        },
    ]

    tmp_dir = tempfile.mkdtemp()
    for style in styles:
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=400,
            max_words=30,
            **style["params"],
        )
        wc.generate_from_frequencies(word_counts)
        output_path = os.path.join(tmp_dir, f"wordcloud_{style['name']}.png")
        wc.to_file(output_path)
        print(f"  {style['name']}: {output_path}")

    print("\n  常用 colormap 选项:")
    colormaps = ["viridis", "plasma", "inferno", "magma", "cividis",
                 "Set2", "Set3", "Paired", "Dark2", "YlGnBu", "RdYlBu"]
    print(f"    {', '.join(colormaps)}")

    print("\n  其他常用参数:")
    other_params = {
        "max_font_size": "最大字号",
        "min_font_size": "最小字号",
        "prefer_horizontal": "词语水平方向比例 (0-1)",
        "relative_scaling": "词频对字号的影响程度 (0-1)",
        "scale": "计算放大倍数，值越大越清晰",
        "repeat": "是否重复词语以填满空间",
    }
    for param, desc in other_params.items():
        print(f"    {param:<22} {desc}")


def demo_wordcloud_with_matplotlib():
    print("\n" + "=" * 60)
    print("6. 使用 matplotlib 展示词云")
    print("=" * 60)

    text = (
        "Python 数据分析 机器学习 深度学习 人工智能 "
        "自然语言处理 计算机视觉 数据可视化 统计学 "
        "Python Python Python 数据分析 数据分析 机器学习 "
        "深度学习 深度学习 人工智能 人工智能 人工智能 "
    )
    word_counts = Counter(text.split())

    font_path = None
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(fp):
            font_path = fp
            break

    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    )
    wc.generate_from_frequencies(word_counts)

    tmp_dir = tempfile.mkdtemp()
    fig_path = os.path.join(tmp_dir, "wordcloud_matplotlib.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("文本数据分析词云", fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  matplotlib 词云图已保存: {fig_path}")

    print("\n  matplotlib 展示优势:")
    print("    - 可以添加标题、坐标轴标签")
    print("    - 支持子图排列，对比多组词云")
    print("    - 灵活控制图像尺寸和分辨率")
    print("    - 可与其他图表组合展示")


if __name__ == "__main__":
    demo_basic_wordcloud()
    demo_chinese_wordcloud()
    demo_wordcloud_from_frequencies()
    demo_wordcloud_mask()
    demo_wordcloud_styles()
    demo_wordcloud_with_matplotlib()

    print("\n" + "=" * 60)
    print("词云生成 演示完成")
    print("=" * 60)
