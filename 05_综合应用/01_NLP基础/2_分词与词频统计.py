# -*- coding: utf-8 -*-
"""
分词与词频统计

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import jieba
from collections import Counter


def demo_jieba_cut():
    print("=" * 60)
    print("1. jieba 分词基础")
    print("=" * 60)

    text = "自然语言处理是人工智能领域中的重要研究方向"

    print("  jieba.cut() 返回生成器:")
    seg_gen = jieba.cut(text)
    print(f"    类型: {type(seg_gen)}")
    print(f"    结果: {' / '.join(seg_gen)}")

    print("\n  jieba.lcut() 返回列表:")
    seg_list = jieba.lcut(text)
    print(f"    类型: {type(seg_list)}")
    print(f"    结果: {seg_list}")

    print("\n  三种分词模式对比:")
    text2 = "我来到北京清华大学"

    seg_precise = jieba.lcut(text2)
    print(f"    精确模式 (默认): {seg_precise}")

    seg_full = jieba.lcut(text2, cut_all=True)
    print(f"    全模式:          {seg_full}")

    seg_search = jieba.lcut_for_search(text2)
    print(f"    搜索引擎模式:    {seg_search}")

    print("\n  搜索引擎模式适用于搜索引擎建立索引，会输出更细粒度的词")


def demo_custom_dict():
    print("\n" + "=" * 60)
    print("2. 自定义词典")
    print("=" * 60)

    text = "深度学习在自然语言处理领域取得了突破性进展"

    print(f"  原始文本: {text}")
    print(f"  默认分词: {jieba.lcut(text)}")

    jieba.add_word("自然语言处理")
    print(f"  添加词后: {jieba.lcut(text)}")

    jieba.add_word("深度学习")
    print(f"  再添加后: {jieba.lcut(text)}")

    print("\n  使用用户词典文件 (jieba.load_userdict):")
    print("""
    # userdict.txt 内容格式: 词语 词频 词性
    自然语言处理 5 n
    深度学习 5 n
    机器学习 5 n

    # 加载方式:
    jieba.load_userdict("userdict.txt")
    """)

    print("  jieba.del_word() 删除词:")
    jieba.del_word("自然语言处理")
    print(f"  删除后分词: {jieba.lcut(text)}")

    jieba.add_word("自然语言处理")


def demo_word_frequency():
    print("\n" + "=" * 60)
    print("3. 词频统计 (Counter)")
    print("=" * 60)

    text = (
        "数据分析是当今最重要的技能之一。"
        "数据分析可以帮助企业做出更好的决策。"
        "学习数据分析需要掌握统计学和编程技能。"
        "数据分析师在就业市场上非常受欢迎。"
    )

    words = jieba.lcut(text)
    words_filtered = [w for w in words if len(w) > 1]
    print(f"  原始分词数: {len(words)}")
    print(f"  过滤单字后: {len(words_filtered)}")

    word_counts = Counter(words_filtered)
    print(f"\n  词频统计 (前10):")
    for word, count in word_counts.most_common(10):
        print(f"    {word}: {count}")

    print(f"\n  总词种数: {len(word_counts)}")
    print(f"  总词次: {sum(word_counts.values())}")

    print(f"\n  Counter 常用方法:")
    print(f"    most_common(3): {word_counts.most_common(3)}")
    print(f"    ['数据分析']: {word_counts['数据分析']}")
    print(f"    elements() 示例: {list(Counter('abracadabra').elements())[:10]}...")


def demo_freq_distribution():
    print("\n" + "=" * 60)
    print("4. 频率分布 (NLTK FreqDist 概念)")
    print("=" * 60)

    text = (
        "机器学习是人工智能的核心技术之一。"
        "深度学习是机器学习的一个重要分支。"
        "自然语言处理和计算机视觉是人工智能的两大应用方向。"
        "强化学习也是机器学习的一种重要方法。"
    )

    words = jieba.lcut(text)
    words_filtered = [w for w in words if len(w) > 1]

    word_counts = Counter(words_filtered)
    total = sum(word_counts.values())

    print("  词频与频率分布:")
    print(f"  {'词语':<12} {'频次':<6} {'频率':<8} {'柱状图'}")
    print("  " + "-" * 50)
    for word, count in word_counts.most_common(10):
        freq = count / total
        bar = "█" * int(freq * 50)
        print(f"  {word:<12} {count:<6} {freq:<8.2%} {bar}")

    print("\n  NLTK FreqDist 概念 (需安装 nltk):")
    print("""
    from nltk import FreqDist

    fdist = FreqDist(words_filtered)
    print(fdist.most_common(10))
    print(fdist.freq('机器学习'))  # 频率
    fdist.plot(10)  # 绘制频率分布图
    """)


def demo_top_n_words():
    print("\n" + "=" * 60)
    print("5. Top-N 高频词提取")
    print("=" * 60)

    texts = [
        "人工智能技术正在深刻改变我们的生活方式和工作模式",
        "深度学习模型在图像识别和语音识别方面取得了显著成果",
        "自然语言处理技术使得机器能够理解和生成人类语言",
        "机器学习算法广泛应用于推荐系统、搜索排序等场景",
        "计算机视觉技术让机器具备了看懂世界的能力",
    ]

    all_words = []
    for text in texts:
        words = jieba.lcut(text)
        words_filtered = [w for w in words if len(w) > 1]
        all_words.extend(words_filtered)

    word_counts = Counter(all_words)

    print("  Top-10 高频词:")
    for i, (word, count) in enumerate(word_counts.most_common(10), 1):
        print(f"    {i:>2}. {word}: {count}")

    print("\n  按文档频率统计 (词出现在多少个文档中):")
    doc_freq = Counter()
    for text in texts:
        words = set(jieba.lcut(text))
        words_filtered = {w for w in words if len(w) > 1}
        for word in words_filtered:
            doc_freq[word] += 1

    print("  文档频率 Top-5:")
    for word, count in doc_freq.most_common(5):
        print(f"    {word}: 出现在 {count}/{len(texts)} 个文档中")


def demo_jieba_tuning():
    print("\n" + "=" * 60)
    print("6. jieba 分词调优技巧")
    print("=" * 60)

    print("  调整词频 (suggest_freq):")
    text = "如果放到post中将出错"
    print(f"    原始: {jieba.lcut(text)}")
    jieba.suggest_freq(("中", "将"), tune=True)
    print(f"    调整后: {jieba.lcut(text)}")

    print("\n  分词耗时对比:")
    import time

    long_text = "自然语言处理是人工智能领域中的重要研究方向。" * 100

    start = time.time()
    jieba.lcut(long_text)
    t1 = time.time() - start

    start = time.time()
    jieba.lcut(long_text, HMM=False)
    t2 = time.time() - start

    print(f"    默认模式 (HMM=True):  {t1*1000:.2f} ms")
    print(f"    关闭HMM (HMM=False):  {t2*1000:.2f} ms")

    print("\n  并行分词 (适用于大规模文本):")
    print("""
    jieba.enable_parallel(4)  # 开启并行分词，参数为并行进程数
    result = jieba.lcut(long_text)
    jieba.disable_parallel()  # 关闭并行分词
    """)


if __name__ == "__main__":
    demo_jieba_cut()
    demo_custom_dict()
    demo_word_frequency()
    demo_freq_distribution()
    demo_top_n_words()
    demo_jieba_tuning()

    print("\n" + "=" * 60)
    print("分词与词频统计 演示完成")
    print("=" * 60)
