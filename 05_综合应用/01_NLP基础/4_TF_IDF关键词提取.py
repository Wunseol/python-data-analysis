# -*- coding: utf-8 -*-
"""
TF-IDF 关键词提取

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import jieba
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def demo_tfidf_basic():
    print("=" * 60)
    print("1. TF-IDF 基本概念")
    print("=" * 60)

    print("  TF (Term Frequency): 词频，词在文档中出现的频率")
    print("  IDF (Inverse Document Frequency): 逆文档频率，衡量词的区分度")
    print("  TF-IDF = TF × IDF，值越大说明词越重要")
    print()
    print("  核心思想:")
    print("    - 一个词在某文档中出现次数越多 (TF高)，对该文档越重要")
    print("    - 一个词在越少的文档中出现 (IDF高)，区分度越高")
    print("    - 常见词 (如'的','是') 虽然TF高但IDF低，TF-IDF值不高")

    docs = ["苹果公司发布了新款手机", "今天吃了一个苹果", "手机行业竞争激烈"]
    print(f"\n  示例文档:")
    for i, doc in enumerate(docs, 1):
        print(f"    文档{i}: {doc}")
    print(f"\n  '苹果' 在文档1中TF高且只出现在2个文档中 → TF-IDF较高")
    print(f"  '手机' 在文档1和3中都出现 → IDF较低 → TF-IDF相对较低")


def demo_tfidf_vectorizer():
    print("\n" + "=" * 60)
    print("2. sklearn TfidfVectorizer 使用")
    print("=" * 60)

    documents = [
        "人工智能技术正在改变世界，深度学习取得了突破性进展",
        "机器学习算法在推荐系统中广泛应用，效果显著",
        "自然语言处理让计算机能够理解和生成人类语言",
        "计算机视觉技术在自动驾驶领域发挥重要作用",
        "深度学习和机器学习是人工智能的核心技术",
    ]

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
        max_features=100,
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()
    print(f"  特征词数量: {len(feature_names)}")
    print(f"  矩阵形状: {tfidf_matrix.shape}")
    print(f"  矩阵类型: {type(tfidf_matrix)}")
    print(f"  前15个特征词: {list(feature_names[:15])}")

    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=[f"文档{i+1}" for i in range(len(documents))],
    )
    print(f"\n  TF-IDF 矩阵 (前5列):")
    print(df.iloc[:, :5].to_string(float_format="{:.4f}".format))


def demo_extract_keywords():
    print("\n" + "=" * 60)
    print("3. 提取每篇文档的 Top 关键词")
    print("=" * 60)

    documents = [
        "人工智能技术正在改变世界，深度学习取得了突破性进展",
        "机器学习算法在推荐系统中广泛应用，效果显著",
        "自然语言处理让计算机能够理解和生成人类语言",
        "计算机视觉技术在自动驾驶领域发挥重要作用",
        "深度学习和机器学习是人工智能的核心技术",
    ]

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    top_n = 5
    print(f"  每篇文档 Top-{top_n} 关键词:\n")
    for doc_idx in range(len(documents)):
        scores = tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        print(f"  文档{doc_idx + 1}: {documents[doc_idx][:30]}...")
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                print(f"    {rank}. {feature_names[idx]:<8} TF-IDF={scores[idx]:.4f}")
        print()


def demo_compare_documents():
    print("=" * 60)
    print("4. 文档相似度比较 (基于 TF-IDF)")
    print("=" * 60)

    documents = [
        "深度学习在图像识别领域取得了巨大成功",
        "深度学习模型在图像分类任务中表现优异",
        "今天天气很好适合出去散步",
    ]

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(tfidf_matrix)

    labels = [f"文档{i+1}" for i in range(len(documents))]
    df_sim = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    print("  文档间余弦相似度:")
    print(df_sim.to_string(float_format="{:.4f}".format))

    print(f"\n  文档1: {documents[0]}")
    print(f"  文档2: {documents[1]}")
    print(f"  文档3: {documents[2]}")
    print(f"\n  文档1与文档2相似度: {similarity_matrix[0][1]:.4f} (内容相近)")
    print(f"  文档1与文档3相似度: {similarity_matrix[0][2]:.4f} (内容无关)")


def demo_tfidf_params():
    print("\n" + "=" * 60)
    print("5. TfidfVectorizer 参数调优")
    print("=" * 60)

    documents = [
        "人工智能技术正在改变世界，深度学习取得了突破性进展",
        "机器学习算法在推荐系统中广泛应用，效果显著",
        "自然语言处理让计算机能够理解和生成人类语言",
        "计算机视觉技术在自动驾驶领域发挥重要作用",
        "深度学习和机器学习是人工智能的核心技术",
    ]

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    print("  常用参数说明:")
    params = {
        "max_features": "最大特征数，保留TF-IDF值最高的N个词",
        "min_df": "最小文档频率，出现次数少于该值的词被忽略",
        "max_df": "最大文档频率，出现在超过该比例文档中的词被忽略",
        "ngram_range": "n-gram范围，如(1,2)包含单字和双字组合",
        "sublinear_tf": "使用1+log(tf)代替tf，缓解高频词影响",
        "norm": "归一化方式，'l2'(默认)或None",
        "use_idf": "是否使用IDF，False则退化为TF",
        "smooth_idf": "IDF平滑，防止除零错误",
    }
    for param, desc in params.items():
        print(f"    {param:<18} {desc}")

    print("\n  参数对比实验:")
    configs = [
        {"max_features": 20, "min_df": 1},
        {"max_features": 50, "min_df": 1, "max_df": 0.8},
        {"max_features": 50, "min_df": 1, "sublinear_tf": True},
    ]
    for config in configs:
        vec = TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            token_pattern=None,
            **config,
        )
        matrix = vec.fit_transform(documents)
        features = vec.get_feature_names_out()
        print(f"    配置 {config} → 特征数: {len(features)}")


def demo_jieba_tfidf():
    print("\n" + "=" * 60)
    print("6. jieba 内置 TF-IDF 关键词提取")
    print("=" * 60)

    import jieba.analyse

    text = (
        "自然语言处理是人工智能领域中的一个重要方向，"
        "它研究如何让计算机理解和生成人类语言。"
        "深度学习技术在自然语言处理中取得了突破性进展，"
        "使得机器翻译、文本分类、情感分析等任务的性能大幅提升。"
    )

    print(f"  原文: {text}")

    keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
    print(f"\n  TF-IDF 提取关键词 (带权重):")
    for word, weight in keywords:
        print(f"    {word:<8} 权重: {weight:.4f}")

    keywords_textrank = jieba.analyse.textrank(text, topK=10, withWeight=True)
    print(f"\n  TextRank 提取关键词 (带权重):")
    for word, weight in keywords_textrank:
        print(f"    {word:<8} 权重: {weight:.4f}")

    print("\n  TF-IDF vs TextRank 对比:")
    tfidf_words = [w for w, _ in jieba.analyse.extract_tags(text, topK=5)]
    textrank_words = [w for w, _ in jieba.analyse.textrank(text, topK=5)]
    common = set(tfidf_words) & set(textrank_words)
    print(f"    TF-IDF Top5: {tfidf_words}")
    print(f"    TextRank Top5: {textrank_words}")
    print(f"    共同关键词: {common}")


if __name__ == "__main__":
    demo_tfidf_basic()
    demo_tfidf_vectorizer()
    demo_extract_keywords()
    demo_compare_documents()
    demo_tfidf_params()
    demo_jieba_tfidf()

    print("\n" + "=" * 60)
    print("TF-IDF 关键词提取 演示完成")
    print("=" * 60)
