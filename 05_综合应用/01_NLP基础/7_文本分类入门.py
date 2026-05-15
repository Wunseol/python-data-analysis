# -*- coding: utf-8 -*-
"""
文本分类入门

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def build_sample_data():
    print("=" * 60)
    print("1. 构建示例文本分类数据")
    print("=" * 60)

    data = {
        "text": [
            "人工智能技术在医疗领域取得重大突破",
            "深度学习模型在图像识别中表现优异",
            "自然语言处理技术助力智能客服发展",
            "机器学习算法优化了推荐系统的效果",
            "计算机视觉技术在自动驾驶中广泛应用",
            "强化学习在游戏AI中取得惊人成绩",
            "神经网络模型在语音识别领域不断进步",
            "知识图谱技术推动智能搜索升级",
            "今日股市大涨，科技股领涨市场",
            "央行宣布降息，刺激经济增长",
            "多家上市公司发布年度财报，利润增长",
            "基金投资策略调整，关注新兴市场",
            "银行推出新的理财产品，收益率可观",
            "A股市场成交量创近期新高",
            "债券市场走势稳健，投资者信心增强",
            "人民币汇率波动，外汇市场关注",
            "中国女排在比赛中展现了顽强拼搏精神",
            "足球世界杯预选赛激烈进行中",
            "NBA季后赛精彩对决，球迷热情高涨",
            "奥运会筹备工作稳步推进",
            "马拉松赛事在全国各地火热开展",
            "乒乓球世锦赛中国队再创佳绩",
            "游泳健将打破世界纪录",
            "羽毛球公开赛精彩纷呈",
        ],
        "category": [
            "科技", "科技", "科技", "科技", "科技", "科技", "科技", "科技",
            "财经", "财经", "财经", "财经", "财经", "财经", "财经", "财经",
            "体育", "体育", "体育", "体育", "体育", "体育", "体育", "体育",
        ],
    }

    df = pd.DataFrame(data)
    print(f"  数据集大小: {df.shape}")
    print(f"  类别分布:")
    for cat, count in df["category"].value_counts().items():
        print(f"    {cat}: {count}条")
    print(f"\n  数据预览:")
    print(df.head(6).to_string(index=False))
    return df


def demo_pipeline():
    print("\n" + "=" * 60)
    print("2. TfidfVectorizer + MultinomialNB 流水线")
    print("=" * 60)

    df = build_sample_data()

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            token_pattern=None,
            max_features=200,
        )),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")
    print(f"  训练集类别分布: {dict(y_train.value_counts())}")

    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"\n  测试集准确率: {score:.4f}")

    return pipeline, X_test, y_test


def demo_evaluation(pipeline, X_test, y_test):
    print("\n" + "=" * 60)
    print("3. 分类评估指标")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)

    print("  分类报告 (classification_report):")
    report = classification_report(y_test, y_pred)
    print(report)

    print("  混淆矩阵 (confusion_matrix):")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    print("\n  指标说明:")
    print("    precision (精确率): 预测为正的样本中，实际为正的比例")
    print("    recall (召回率): 实际为正的样本中，被正确预测的比例")
    print("    f1-score: 精确率和召回率的调和平均")
    print("    support: 每个类别的样本数")


def demo_predict(pipeline):
    print("\n" + "=" * 60)
    print("4. 预测新文本")
    print("=" * 60)

    new_texts = [
        "深度学习在医疗影像诊断中取得重大进展",
        "股市震荡调整，投资者需谨慎操作",
        "中国乒乓球队在国际赛事中再创辉煌",
        "5G技术推动物联网产业快速发展",
        "央行发布最新货币政策报告",
        "田径世锦赛中国队表现出色",
    ]

    predictions = pipeline.predict(new_texts)
    probabilities = pipeline.predict_proba(new_texts)
    classes = pipeline.classes_

    print(f"  类别顺序: {list(classes)}")
    print(f"\n  预测结果:")
    for text, pred, probs in zip(new_texts, predictions, probabilities):
        prob_str = ", ".join([f"{c}:{p:.3f}" for c, p in zip(classes, probs)])
        print(f"    '{text[:20]}...' → {pred} ({prob_str})")


def demo_feature_names(pipeline):
    print("\n" + "=" * 60)
    print("5. 特征词分析")
    print("=" * 60)

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    print(f"  特征词总数: {len(feature_names)}")
    print(f"  前20个特征词: {list(feature_names[:20])}")

    print(f"\n  各类别最重要的特征词 (朴素贝叶斯对数概率):")
    for i, category in enumerate(clf.classes_):
        top_indices = clf.feature_log_prob_[i].argsort()[::-1][:8]
        top_words = [(feature_names[idx], clf.feature_log_prob_[i][idx]) for idx in top_indices]
        words_str = ", ".join([f"{w}({p:.3f})" for w, p in top_words])
        print(f"    {category}: {words_str}")


def demo_cross_validation():
    print("\n" + "=" * 60)
    print("6. 交叉验证评估")
    print("=" * 60)

    from sklearn.model_selection import cross_val_score

    df = build_sample_data()

    def chinese_tokenizer(text):
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            token_pattern=None,
            max_features=200,
        )),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    X = df["text"]
    y = df["category"]

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  5折交叉验证结果:")
    print(f"    各折准确率: {scores}")
    print(f"    平均准确率: {scores.mean():.4f} (±{scores.std():.4f})")

    scores_f1 = cross_val_score(pipeline, X, y, cv=5, scoring="f1_macro")
    print(f"\n  F1-macro 交叉验证:")
    print(f"    各折F1: {scores_f1}")
    print(f"    平均F1: {scores_f1.mean():.4f} (±{scores_f1.std():.4f})")

    print("\n  交叉验证优势:")
    print("    - 充分利用有限数据")
    print("    - 评估结果更稳定可靠")
    print("    - 可以观察模型稳定性 (标准差)")


if __name__ == "__main__":
    pipeline, X_test, y_test = demo_pipeline()
    demo_evaluation(pipeline, X_test, y_test)
    demo_predict(pipeline)
    demo_feature_names(pipeline)
    demo_cross_validation()

    print("\n" + "=" * 60)
    print("文本分类入门 演示完成")
    print("=" * 60)
