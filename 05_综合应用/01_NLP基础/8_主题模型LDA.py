# -*- coding: utf-8 -*-
"""
主题模型 LDA

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def build_sample_data():
    print("=" * 60)
    print("1. 构建示例文本数据")
    print("=" * 60)

    documents = [
        "人工智能技术在医疗影像诊断中取得重大突破，深度学习模型准确率大幅提升",
        "机器学习算法在金融风控领域广泛应用，有效降低了信贷风险",
        "深度学习在自动驾驶领域发挥关键作用，感知系统不断优化",
        "自然语言处理技术推动智能客服发展，用户体验显著改善",
        "计算机视觉技术在工业质检中应用广泛，缺陷检测精度提高",
        "强化学习在机器人控制中取得进展，动作决策更加智能",
        "知识图谱技术助力智能搜索，信息检索更加精准",
        "神经网络模型在语音识别领域持续进步，识别率不断提升",
        "央行宣布降息政策，刺激经济增长，市场信心增强",
        "A股市场大幅上涨，科技板块领涨，成交量创近期新高",
        "上市公司发布年度财报，多家企业利润实现两位数增长",
        "基金投资策略调整，机构投资者看好新兴市场前景",
        "银行推出新型理财产品，预期收益率具有吸引力",
        "债券市场走势稳健，国债收益率曲线趋于平坦",
        "人民币汇率双向波动，外汇市场保持基本稳定",
        "保险行业数字化转型加速，线上投保比例持续上升",
        "中国女排在世界联赛中表现出色，展现了顽强拼搏精神",
        "足球世界杯预选赛竞争激烈，多支强队争夺出线名额",
        "NBA季后赛精彩对决，多位球星展现出色竞技状态",
        "奥运会筹备工作稳步推进，各项测试赛顺利举行",
        "马拉松赛事在全国各地火热开展，参与人数创新高",
        "乒乓球世锦赛中国队包揽多项冠军，实力碾压对手",
        "游泳健将在国际赛事中打破世界纪录，创造历史",
        "羽毛球公开赛精彩纷呈，多场比赛打满三局",
    ]

    print(f"  文档数量: {len(documents)}")
    print(f"  示例文档:")
    for i, doc in enumerate(documents[:3], 1):
        print(f"    {i}. {doc[:40]}...")

    return documents


def demo_lda_basic():
    print("\n" + "=" * 60)
    print("2. LDA 主题模型基础")
    print("=" * 60)

    documents = build_sample_data()

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    print("  LDA (Latent Dirichlet Allocation) 核心概念:")
    print("    - 无监督学习，自动发现文档集中的隐含主题")
    print("    - 每个主题由一组词语的概率分布表示")
    print("    - 每篇文档由多个主题的混合分布表示")
    print("    - 关键参数: n_components (主题数量)")

    vectorizer = CountVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
        max_features=200,
        max_df=0.9,
        min_df=2,
    )

    dtm = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n  文档-词矩阵 (DTM):")
    print(f"    形状: {dtm.shape} (文档数 × 特征词数)")
    print(f"    稀疏度: {1 - dtm.nnz / (dtm.shape[0] * dtm.shape[1]):.2%}")

    n_topics = 3
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        random_state=42,
        learning_method="batch",
    )

    lda.fit(dtm)
    print(f"\n  LDA 模型参数:")
    print(f"    主题数: {lda.n_components}")
    print(f"    迭代次数: {lda.max_iter}")
    print(f"    学习方法: {lda.learning_method}")
    print(f"    困惑度: {lda.perplexity(dtm):.2f}")

    return lda, vectorizer, feature_names, documents


def demo_topic_words(lda, feature_names):
    print("\n" + "=" * 60)
    print("3. 主题-词语分布")
    print("=" * 60)

    n_top_words = 8

    print(f"  每个主题的 Top-{n_top_words} 词语:\n")
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        print(f"  主题 {topic_idx}:")
        for word, weight in zip(top_words, top_weights):
            print(f"    {word:<10} 权重: {weight:.2f}")
        print()

    print("  主题解读 (根据高频词判断):")
    topic_labels = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:5]
        top_words = [feature_names[i] for i in top_indices]
        print(f"    主题 {topic_idx}: {', '.join(top_words)}")
        if any(w in "深度学习神经网络模型算法" for w in top_words):
            topic_labels[topic_idx] = "科技"
        elif any(w in "股市基金投资市场" for w in top_words):
            topic_labels[topic_idx] = "财经"
        elif any(w in "比赛赛事冠军球队" for w in top_words):
            topic_labels[topic_idx] = "体育"
        else:
            topic_labels[topic_idx] = f"未知主题{topic_idx}"
    print(f"    主题标签: {topic_labels}")
    return topic_labels


def demo_doc_topic(lda, vectorizer, documents, topic_labels):
    print("\n" + "=" * 60)
    print("4. 文档-主题分布")
    print("=" * 60)

    dtm = vectorizer.transform(documents)
    doc_topic_dist = lda.transform(dtm)

    print(f"  文档-主题分布矩阵形状: {doc_topic_dist.shape}")
    print(f"  每行和为1 (表示文档在各主题上的概率分配)")

    print(f"\n  各文档的主题分布 (前8篇):")
    for i in range(min(8, len(documents))):
        dominant_topic = doc_topic_dist[i].argmax()
        probs = doc_topic_dist[i]
        prob_str = " | ".join([f"主题{t}:{p:.3f}" for t, p in enumerate(probs)])
        label = topic_labels.get(dominant_topic, f"主题{dominant_topic}")
        print(f"  文档{i+1}: {documents[i][:30]}...")
        print(f"    分布: [{prob_str}] → 主要主题: {label}")

    print(f"\n  主题分布统计:")
    dominant_topics = doc_topic_dist.argmax(axis=1)
    for t in range(lda.n_components):
        count = (dominant_topics == t).sum()
        label = topic_labels.get(t, f"主题{t}")
        print(f"    {label}: {count}篇文档 ({count / len(documents):.1%})")


def demo_perplexity():
    print("\n" + "=" * 60)
    print("5. 主题数选择 (困惑度)")
    print("=" * 60)

    documents = build_sample_data()

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    vectorizer = CountVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
        max_features=200,
        max_df=0.9,
        min_df=2,
    )
    dtm = vectorizer.fit_transform(documents)

    print("  不同主题数的困惑度对比:")
    print(f"  {'主题数':<8} {'困惑度':<12} {'对数似然':<12}")
    print("  " + "-" * 32)

    results = []
    for n_topics in [2, 3, 4, 5, 6]:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=20,
            random_state=42,
            learning_method="batch",
        )
        lda.fit(dtm)
        perplexity = lda.perplexity(dtm)
        score = lda.score(dtm)
        results.append((n_topics, perplexity, score))
        print(f"  {n_topics:<8} {perplexity:<12.2f} {score:<12.2f}")

    print("\n  选择建议:")
    print("    - 困惑度越低，模型对数据的拟合越好")
    print("    - 但主题数过多会导致过拟合，主题不具可解释性")
    print("    - 应结合困惑度和主题可解释性综合判断")
    print("    - 也可使用对数似然 (score) 越大越好")


def demo_pyldavis():
    print("\n" + "=" * 60)
    print("6. pyLDAvis 可视化 (概念)")
    print("=" * 60)

    print("  pyLDAvis 是 LDA 主题模型的可视化工具，提供交互式探索:")
    print()
    print("  安装: pip install pyldavis")
    print()
    print("  使用示例:")
    print("""
    import pyLDAvis
    from pyLDAvis import sklearn as sklearn_vis

    # 创建可视化面板
    panel = sklearn_vis.prepare(lda, dtm, vectorizer)

    # 保存为 HTML 文件
    pyLDAvis.save_html(panel, "lda_vis.html")

    # 在 Jupyter Notebook 中直接显示
    # panel
    """)
    print("  可视化面板包含:")
    print("    - 左侧: 主题间距离图 (降维后的二维散点图)")
    print("    - 右侧: 选中主题的关键词及其频率")
    print("    - λ滑块: 调节词语相关性计算方式")
    print()
    print("  注意: pyLDAvis 与 sklearn 版本可能存在兼容性问题")
    print("  替代方案: 使用 matplotlib 手动绘制主题-词语热力图")

    documents = build_sample_data()

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    vectorizer = CountVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
        max_features=100,
        max_df=0.9,
        min_df=2,
    )
    dtm = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=3, max_iter=20, random_state=42, learning_method="batch",
    )
    lda.fit(dtm)

    n_top = 10
    topic_word_matrix = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

    print(f"\n  主题-词语热力图 (Top {n_top} 词语概率):")
    top_indices = lda.components_.sum(axis=0).argsort()[::-1][:n_top]
    top_features = [feature_names[i] for i in top_indices]

    header = f"  {'词语':<10}" + "".join([f"{'主题'+str(i):<10}" for i in range(3)])
    print(header)
    print("  " + "-" * 40)
    for idx in top_indices:
        row = f"  {feature_names[idx]:<10}"
        for t in range(3):
            row += f"{topic_word_matrix[t, idx]:<10.4f}"
        print(row)


if __name__ == "__main__":
    lda, vectorizer, feature_names, documents = demo_lda_basic()
    topic_labels = demo_topic_words(lda, feature_names)
    demo_doc_topic(lda, vectorizer, documents, topic_labels)
    demo_perplexity()
    demo_pyldavis()

    print("\n" + "=" * 60)
    print("主题模型 LDA 演示完成")
    print("=" * 60)
