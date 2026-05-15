# -*- coding: utf-8 -*-
"""
综合案例: 新闻文本分析

数据来源: 脚本内自建新闻文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import re
import os
import tempfile
import jieba
import jieba.analyse
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def build_news_data():
    print("=" * 60)
    print("步骤1: 构建新闻文本数据集")
    print("=" * 60)

    news_data = {
        "title": [
            "AI芯片技术取得重大突破",
            "深度学习模型在医疗诊断中表现优异",
            "5G技术推动物联网产业快速发展",
            "量子计算研究取得新进展",
            "自动驾驶技术进入商业化阶段",
            "央行降息刺激经济增长",
            "A股市场创年内新高",
            "基金投资策略转向价值投资",
            "银行理财产品收益率上升",
            "上市公司年报显示利润增长",
            "数字货币监管政策出台",
            "保险行业数字化转型加速",
            "中国女排世界联赛夺冠",
            "NBA全明星赛精彩纷呈",
            "奥运会新增项目引发关注",
            "马拉松运动在全国蓬勃发展",
            "足球青训体系建设加速推进",
            "电竞产业规模持续扩大",
        ],
        "content": [
            "人工智能芯片技术取得重大突破，新一代AI芯片算力提升三倍，能效比显著改善。深度学习推理速度大幅提高，为边缘计算和终端设备带来新的可能。多家科技企业竞相布局AI芯片赛道，产业竞争日趋激烈。",
            "深度学习模型在医疗影像诊断中表现优异，肺结节检测准确率达到95%以上。该技术已在多家三甲医院投入临床使用，辅助医生进行疾病诊断。研究团队表示将继续优化模型，提升诊断效率。",
            "5G技术推动物联网产业快速发展，智能家居、工业互联网等应用场景不断拓展。5G网络覆盖范围持续扩大，为万物互联提供坚实基础。运营商加大5G基站建设投入，用户规模稳步增长。",
            "量子计算研究取得新进展，研究团队成功实现更多量子比特的稳定操控。量子计算在密码学、材料科学等领域展现巨大潜力。各国加大量子计算研究投入，竞争格局日趋激烈。",
            "自动驾驶技术进入商业化阶段，多家企业获得无人驾驶测试牌照。L4级自动驾驶车辆在限定区域开展运营服务，安全记录良好。政策法规逐步完善，为自动驾驶商业化铺平道路。",
            "央行宣布降息政策，下调基准利率25个基点，旨在刺激经济增长。市场分析人士认为降息将降低企业融资成本，提振投资信心。房地产市场和消费市场有望受益于宽松货币政策。",
            "A股市场今日大幅上涨，沪指突破年内高点，科技板块领涨市场。两市成交量突破万亿，创近期新高。分析师认为市场情绪回暖，资金面宽松推动行情向上。投资者需注意控制风险。",
            "基金投资策略转向价值投资，多家机构看好低估值蓝筹股。成长股估值回调后配置价值凸显，市场风格切换迹象明显。基金经理建议投资者关注基本面，坚持长期投资理念。",
            "银行理财产品收益率上升，多款产品预期年化收益率超过4%。理财市场回暖吸引大量资金涌入，产品发行规模持续增长。投资者应根据自身风险偏好选择合适的产品。",
            "上市公司年报显示利润增长，超过七成企业实现盈利增长。科技和消费行业表现突出，多家公司营收创历史新高。分析师预计经济复苏将推动企业盈利持续改善。",
            "数字货币监管政策出台，明确数字人民币的法定货币地位。政策规范了数字货币交易行为，加强投资者保护。金融机构积极布局数字货币相关业务，市场发展前景广阔。",
            "保险行业数字化转型加速，线上投保比例超过60%。人工智能技术在核保理赔环节广泛应用，效率大幅提升。保险科技企业获得大量融资，推动行业创新升级。",
            "中国女排在世界联赛中表现出色，以全胜战绩夺得冠军。队员们在比赛中展现了顽强拼搏的精神和精湛的技术。主教练对球队表现给予高度评价，期待在奥运会上再创佳绩。",
            "NBA全明星赛精彩纷呈，多位球星展现出色竞技状态。三分球大赛和扣篮大赛备受球迷关注，现场气氛热烈。联盟表示将继续推动篮球运动在全球的发展。",
            "奥运会新增项目引发广泛关注，滑板、攀岩等年轻化项目吸引大量观众。国际奥委会表示新增项目有助于吸引年轻观众，推动奥林匹克运动发展。运动员们积极备战新项目。",
            "马拉松运动在全国蓬勃发展，全年赛事超过千场，参与人数突破千万。马拉松产业链不断完善，带动运动装备、旅游等相关产业发展。跑者水平持续提升，多项赛会纪录被打破。",
            "足球青训体系建设加速推进，多地建设专业足球训练基地。青少年足球人口持续增长，人才培养体系逐步完善。足协出台多项政策支持青训发展，为国家队输送人才。",
            "电竞产业规模持续扩大，全球观众人数突破五亿。中国电竞战队在国际赛事中屡获佳绩，产业生态日趋成熟。电竞正式成为亚运会比赛项目，行业发展迎来新机遇。",
        ],
        "category": [
            "科技", "科技", "科技", "科技", "科技",
            "财经", "财经", "财经", "财经", "财经", "财经", "财经",
            "体育", "体育", "体育", "体育", "体育", "体育",
        ],
    }

    df = pd.DataFrame(news_data)
    print(f"  数据集大小: {df.shape}")
    print(f"  类别分布:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat}: {count}条")
    return df


class TextCleaner:
    def __init__(self):
        self.stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "它", "们", "那", "些", "什么", "怎么", "如何", "可以", "因为",
            "所以", "但是", "而且", "或者", "如果", "虽然", "已经", "还是",
            "这个", "那个", "之", "与", "及", "等", "被", "把", "让", "给",
            "从", "向", "对", "为", "以", "而", "又", "并", "且", "则",
        }

    def clean(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        text = self.clean(text)
        words = jieba.lcut(text)
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def tokenize_str(self, text):
        return " ".join(self.tokenize(text))


def step2_cleaning(df):
    print("\n" + "=" * 60)
    print("步骤2: 文本清洗与分词")
    print("=" * 60)

    cleaner = TextCleaner()

    print("  清洗前:")
    print(f"    {df['content'].iloc[0][:60]}...")

    cleaned = cleaner.clean(df['content'].iloc[0])
    print(f"\n  清洗后:")
    print(f"    {cleaned[:60]}...")

    tokenized = cleaner.tokenize(df['content'].iloc[0])
    print(f"\n  分词结果 (前15个):")
    print(f"    {tokenized[:15]}")

    df['cleaned_content'] = df['content'].apply(cleaner.clean)
    df['tokens'] = df['content'].apply(cleaner.tokenize)
    df['token_str'] = df['content'].apply(cleaner.tokenize_str)

    print(f"\n  各文档分词数量统计:")
    df['token_count'] = df['tokens'].apply(len)
    for cat in df['category'].unique():
        avg = df[df['category'] == cat]['token_count'].mean()
        print(f"    {cat}: 平均 {avg:.1f} 个词")

    return df, cleaner


def step3_keywords(df):
    print("\n" + "=" * 60)
    print("步骤3: TF-IDF 关键词提取")
    print("=" * 60)

    cleaner = TextCleaner()

    vectorizer = TfidfVectorizer(
        tokenizer=cleaner.tokenize,
        token_pattern=None,
        max_features=200,
    )
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    feature_names = vectorizer.get_feature_names_out()

    print(f"  特征词数量: {len(feature_names)}")

    print(f"\n  每篇新闻 Top-5 关键词:")
    for idx in range(min(6, len(df))):
        scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = scores.argsort()[::-1][:5]
        keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        kw_str = ", ".join([f"{w}({s:.3f})" for w, s in keywords])
        print(f"    [{df['category'].iloc[idx]}] {df['title'].iloc[idx]}")
        print(f"      关键词: {kw_str}")

    print(f"\n  各类别高频关键词:")
    for cat in df['category'].unique():
        cat_texts = df[df['category'] == cat]['content']
        all_words = []
        for text in cat_texts:
            all_words.extend(cleaner.tokenize(text))
        word_counts = Counter(all_words)
        top5 = word_counts.most_common(5)
        print(f"    {cat}: {', '.join([f'{w}({c})' for w, c in top5])}")

    return vectorizer, tfidf_matrix, feature_names


def step4_topic_model(df):
    print("\n" + "=" * 60)
    print("步骤4: LDA 主题建模")
    print("=" * 60)

    cleaner = TextCleaner()

    vectorizer = CountVectorizer(
        tokenizer=cleaner.tokenize,
        token_pattern=None,
        max_features=200,
        max_df=0.9,
        min_df=2,
    )
    dtm = vectorizer.fit_transform(df['content'])
    feature_names = vectorizer.get_feature_names_out()

    n_topics = 3
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=30,
        random_state=42,
        learning_method="batch",
    )
    lda.fit(dtm)

    print(f"  主题数: {n_topics}")
    print(f"  困惑度: {lda.perplexity(dtm):.2f}")

    print(f"\n  各主题 Top-8 词语:")
    topic_labels = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:8]
        top_words = [feature_names[i] for i in top_indices]
        print(f"    主题 {topic_idx}: {', '.join(top_words)}")

        if any(w in "技术模型芯片计算" for w in top_words):
            topic_labels[topic_idx] = "科技"
        elif any(w in "市场投资增长利率" for w in top_words):
            topic_labels[topic_idx] = "财经"
        elif any(w in "比赛赛事运动训练" for w in top_words):
            topic_labels[topic_idx] = "体育"
        else:
            topic_labels[topic_idx] = f"主题{topic_idx}"

    print(f"\n  主题标签: {topic_labels}")

    doc_topic_dist = lda.transform(dtm)
    print(f"\n  文档-主题分布 (前6篇):")
    for i in range(min(6, len(df))):
        dominant = doc_topic_dist[i].argmax()
        probs = doc_topic_dist[i]
        label = topic_labels.get(dominant, f"主题{dominant}")
        prob_str = " | ".join([f"{topic_labels.get(t, f'主题{t}')}: {p:.3f}" for t, p in enumerate(probs)])
        print(f"    {df['title'].iloc[i][:20]}... → {label} [{prob_str}]")

    return lda, vectorizer, topic_labels


def step5_sentiment(df):
    print("\n" + "=" * 60)
    print("步骤5: 情感分析")
    print("=" * 60)

    try:
        from snownlp import SnowNLP
    except ImportError:
        print("  SnowNLP 未安装，请运行: pip install snownlp")
        print("  跳过情感分析步骤")
        return

    df['sentiment_score'] = df['content'].apply(lambda x: SnowNLP(x).sentiments)

    def classify_sentiment(score):
        if score > 0.6:
            return "积极"
        elif score < 0.4:
            return "消极"
        else:
            return "中性"

    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

    print(f"  情感分析结果:")
    for _, row in df.iterrows():
        print(f"    [{row['category']}] {row['title'][:20]}... → {row['sentiment']} ({row['sentiment_score']:.3f})")

    print(f"\n  各类别情感统计:")
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        avg_score = cat_df['sentiment_score'].mean()
        sentiment_dist = cat_df['sentiment'].value_counts().to_dict()
        print(f"    {cat}: 平均分数={avg_score:.3f}, 分布={sentiment_dist}")

    print(f"\n  整体情感分布:")
    for s, count in df['sentiment'].value_counts().items():
        print(f"    {s}: {count}条 ({count / len(df):.1%})")


def step6_classification(df):
    print("\n" + "=" * 60)
    print("步骤6: 文本分类")
    print("=" * 60)

    cleaner = TextCleaner()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=cleaner.tokenize,
            token_pattern=None,
            max_features=200,
        )),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    X = df['content']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"  训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"  测试集准确率: {score:.4f}")

    y_pred = pipeline.predict(X_test)
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred))

    new_texts = [
        "新一代AI大模型在自然语言理解方面取得突破",
        "股市震荡调整，投资者关注政策动向",
        "中国运动员在国际大赛中再创佳绩",
    ]
    predictions = pipeline.predict(new_texts)
    probabilities = pipeline.predict_proba(new_texts)
    classes = pipeline.classes_

    print(f"  新文本预测:")
    for text, pred, probs in zip(new_texts, predictions, probabilities):
        prob_str = ", ".join([f"{c}:{p:.3f}" for c, p in zip(classes, probs)])
        print(f"    '{text[:25]}...' → {pred} ({prob_str})")


def step7_visualization(df):
    print("\n" + "=" * 60)
    print("步骤7: 可视化")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError:
        print("  matplotlib 或 wordcloud 未安装，跳过可视化")
        return

    cleaner = TextCleaner()
    tmp_dir = tempfile.mkdtemp()

    font_path = None
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if os.path.exists(fp):
            font_path = fp
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, cat in enumerate(df['category'].unique()):
        cat_texts = df[df['category'] == cat]['content']
        all_words = []
        for text in cat_texts:
            all_words.extend(cleaner.tokenize(text))
        text_for_cloud = " ".join(all_words)

        wc = WordCloud(
            font_path=font_path,
            width=600,
            height=400,
            background_color="white",
            max_words=50,
            colormap="viridis",
        )
        wc.generate(text_for_cloud)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"{cat}类新闻词云", fontsize=14)
        axes[i].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(tmp_dir, "news_wordcloud.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  词云图已保存: {fig_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cat_counts = df['category'].value_counts()
    axes[0].bar(cat_counts.index, cat_counts.values, color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[0].set_title("各类别新闻数量")
    axes[0].set_ylabel("数量")

    if 'sentiment_score' in df.columns:
        for cat in df['category'].unique():
            cat_scores = df[df['category'] == cat]['sentiment_score']
            axes[1].hist(cat_scores, alpha=0.6, label=cat, bins=8)
        axes[1].set_title("各类别情感分数分布")
        axes[1].set_xlabel("情感分数")
        axes[1].set_ylabel("数量")
        axes[1].legend()
    else:
        df['token_count'] = df['tokens'].apply(len)
        for cat in df['category'].unique():
            cat_counts_tokens = df[df['category'] == cat]['token_count']
            axes[1].hist(cat_counts_tokens, alpha=0.6, label=cat, bins=8)
        axes[1].set_title("各类别分词数量分布")
        axes[1].set_xlabel("词数")
        axes[1].set_ylabel("数量")
        axes[1].legend()

    plt.tight_layout()
    fig_path2 = os.path.join(tmp_dir, "news_analysis.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  分析图已保存: {fig_path2}")


def step8_summary(df):
    print("\n" + "=" * 60)
    print("步骤8: 生成分析摘要报告")
    print("=" * 60)

    cleaner = TextCleaner()

    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║           新闻文本分析摘要报告                   ║")
    print("  ╚══════════════════════════════════════════════════╝")

    print(f"\n  【数据概览】")
    print(f"    新闻总数: {len(df)}")
    print(f"    类别数量: {df['category'].nunique()}")
    print(f"    类别列表: {', '.join(df['category'].unique())}")

    print(f"\n  【文本统计】")
    df['char_count'] = df['content'].apply(len)
    df['token_count'] = df['content'].apply(lambda x: len(cleaner.tokenize(x)))
    print(f"    平均字符数: {df['char_count'].mean():.0f}")
    print(f"    平均词数: {df['token_count'].mean():.0f}")
    print(f"    最短新闻: {df['char_count'].min()} 字符")
    print(f"    最长新闻: {df['char_count'].max()} 字符")

    print(f"\n  【各类别统计】")
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        print(f"    {cat}:")
        print(f"      新闻数: {len(cat_df)}")
        print(f"      平均词数: {cat_df['token_count'].mean():.0f}")

        all_words = []
        for text in cat_df['content']:
            all_words.extend(cleaner.tokenize(text))
        word_counts = Counter(all_words)
        top3 = word_counts.most_common(3)
        print(f"      高频词: {', '.join([f'{w}({c})' for w, c in top3])}")

    if 'sentiment_score' in df.columns:
        print(f"\n  【情感分析】")
        print(f"    整体平均情感分数: {df['sentiment_score'].mean():.3f}")
        for cat in df['category'].unique():
            avg = df[df['category'] == cat]['sentiment_score'].mean()
            print(f"    {cat} 平均情感分数: {avg:.3f}")

    print(f"\n  【分析流程】")
    print(f"    1. 文本加载 → 2. 清洗分词 → 3. 关键词提取")
    print(f"    4. 主题建模 → 5. 情感分析 → 6. 文本分类")
    print(f"    7. 可视化   → 8. 摘要报告")

    print(f"\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║               报告生成完毕                       ║")
    print(f"  ╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    df = build_news_data()
    df, cleaner = step2_cleaning(df)
    vectorizer, tfidf_matrix, feature_names = step3_keywords(df)
    lda, cnt_vectorizer, topic_labels = step4_topic_model(df)
    step5_sentiment(df)
    step6_classification(df)
    step7_visualization(df)
    step8_summary(df)

    print("\n" + "=" * 60)
    print("综合案例: 新闻文本分析 演示完成")
    print("=" * 60)
