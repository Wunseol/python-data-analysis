# -*- coding: utf-8 -*-
"""
停用词过滤与文本清洗

数据来源: 脚本内自建文本数据
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import re
import jieba
from collections import Counter


def build_stopwords():
    print("=" * 60)
    print("1. 构建停用词表")
    print("=" * 60)

    stopwords_cn = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
        "它", "们", "那", "些", "什么", "怎么", "如何", "可以", "因为",
        "所以", "但是", "而且", "或者", "如果", "虽然", "已经", "还是",
        "这个", "那个", "这些", "那些", "之", "与", "及", "等", "被",
        "把", "让", "给", "从", "向", "对", "为", "以", "而", "又",
        "并", "且", "则", "更", "最", "其", "此", "该", "每", "各",
    }

    print(f"  内置停用词数量: {len(stopwords_cn)}")
    print(f"  示例停用词: {list(stopwords_cn)[:10]}")

    print("\n  从文件加载停用词 (常见做法):")
    print("""
    # 常见中文停用词表来源:
    #   - 哈工大停用词表
    #   - 百度停用词表
    #   - 四川大学机器智能实验室停用词库

    def load_stopwords(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    stopwords = load_stopwords("stopwords.txt")
    """)

    return stopwords_cn


def demo_stopword_filtering(stopwords):
    print("\n" + "=" * 60)
    print("2. 停用词过滤")
    print("=" * 60)

    text = "自然语言处理是人工智能领域中一个非常重要的研究方向"
    print(f"  原始文本: {text}")

    words = jieba.lcut(text)
    print(f"  分词结果: {words}")

    words_filtered = [w for w in words if w not in stopwords and len(w) > 1]
    print(f"  过滤停用词: {words_filtered}")

    removed = [w for w in words if w in stopwords or len(w) <= 1]
    print(f"  被移除的词: {removed}")

    print("\n  批量文本停用词过滤:")
    texts = [
        "这个机器学习模型的效果非常好",
        "我们需要更多的数据来训练模型",
        "深度学习是一种非常强大的技术",
    ]
    for t in texts:
        words = jieba.lcut(t)
        filtered = [w for w in words if w not in stopwords and len(w) > 1]
        print(f"    原文: {t}")
        print(f"    过滤: {filtered}")


def demo_regex_cleaning():
    print("\n" + "=" * 60)
    print("3. 正则表达式文本清洗")
    print("=" * 60)

    raw_text = '访问 https://www.example.com 查看更多！电话: 010-12345678，价格￥99.8。<p>HTML标签</p>'
    print(f"  原始文本: {raw_text}")

    cleaned = re.sub(r'https?://\S+', '', raw_text)
    print(f"\n  去除URL: {cleaned.strip()}")

    cleaned = re.sub(r'<[^>]+>', '', raw_text)
    print(f"  去除HTML标签: {cleaned.strip()}")

    cleaned = re.sub(r'[\d\-]+', '', raw_text)
    print(f"  去除数字和连字符: {cleaned.strip()}")

    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', raw_text)
    print(f"  仅保留中文和英文: {cleaned.strip()}")

    cleaned = re.sub(r'[，。！？、；：""''（）【】《》\s]', ' ', raw_text)
    print(f"  中文标点替换为空格: {cleaned.strip()}")

    print("\n  常用正则清洗模式汇总:")
    patterns = {
        r'https?://\S+': 'URL链接',
        r'<[^>]+>': 'HTML标签',
        r'[\u4e00-\u9fa5]': '中文字符范围',
        r'[^\u4e00-\u9fa5a-zA-Z\s]': '非中文非英文非空格',
        r'\s+': '多余空白',
        r'[\d]+': '数字',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': '邮箱',
    }
    for pattern, desc in patterns.items():
        print(f"    {desc:<15} → {pattern}")


def demo_text_normalization():
    print("\n" + "=" * 60)
    print("4. 文本规范化")
    print("=" * 60)

    text = "ＮＬＰ　是　自　然　语　言　处　理　的　缩　写！！！   "
    print(f"  原始文本: '{text}'")

    import unicodedata

    normalized = unicodedata.normalize("NFKC", text)
    print(f"  Unicode规范化: '{normalized}'")

    normalized = normalized.strip()
    print(f"  去除首尾空白: '{normalized}'")

    normalized = re.sub(r'\s+', ' ', normalized)
    print(f"  合并多余空白: '{normalized}'")

    normalized = re.sub(r'[！？。]+$', '。', normalized)
    print(f"  合并重复标点: '{normalized}'")

    print("\n  全角半角转换:")
    fullwidth = "ＡＢＣ１２３"
    halfwidth = unicodedata.normalize("NFKC", fullwidth)
    print(f"    全角: {fullwidth} → 半角: {halfwidth}")

    print("\n  繁简转换 (概念，需安装 opencc):")
    print("""
    # pip install opencc-python-reimplemented
    from opencc import OpenCC
    cc = OpenCC('t2s')  # 繁体转简体
    simplified = cc.convert('自然語言處理')
    """)


def build_cleaning_pipeline(stopwords):
    print("\n" + "=" * 60)
    print("5. 构建文本清洗流水线")
    print("=" * 60)

    class TextCleaner:
        def __init__(self, stopwords=None):
            self.stopwords = stopwords or set()
            self.steps = [
                ("去除URL", self._remove_urls),
                ("去除HTML标签", self._remove_html),
                ("去除特殊字符", self._remove_special_chars),
                ("合并空白", self._normalize_whitespace),
                ("分词", self._tokenize),
                ("过滤停用词", self._filter_stopwords),
                ("过滤短词", self._filter_short),
            ]

        def _remove_urls(self, text):
            return re.sub(r'https?://\S+', '', text)

        def _remove_html(self, text):
            return re.sub(r'<[^>]+>', '', text)

        def _remove_special_chars(self, text):
            return re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', ' ', text)

        def _normalize_whitespace(self, text):
            return re.sub(r'\s+', ' ', text).strip()

        def _tokenize(self, text):
            return jieba.lcut(text)

        def _filter_stopwords(self, words):
            return [w for w in words if w not in self.stopwords]

        def _filter_short(self, words):
            return [w for w in words if len(w) > 1]

        def clean(self, text, verbose=False):
            result = text
            if verbose:
                print(f"    原始: {text}")
            for step_name, step_func in self.steps:
                result = step_func(result)
                if verbose:
                    print(f"    {step_name}: {result}")
            return result

    cleaner = TextCleaner(stopwords=stopwords)

    raw_text = '深度学习是<a href="#">AI</a>的核心！详见 https://example.com ！！！'
    print(f"  原始文本: {raw_text}")
    print("  逐步清洗过程:")
    cleaned = cleaner.clean(raw_text, verbose=True)
    print(f"\n  最终结果: {cleaned}")

    print("\n  批量清洗:")
    texts = [
        '自然语言处理(NLP)是AI的重要方向，参考 https://nlp.example.com',
        '<div>机器学习</div>和深度学习是当前最热门的技术！！！',
        '数据科学家需要掌握Python、SQL和统计学等技能...',
    ]
    for t in texts:
        result = cleaner.clean(t)
        print(f"    原文: {t}")
        print(f"    清洗: {result}")


if __name__ == "__main__":
    stopwords = build_stopwords()
    demo_stopword_filtering(stopwords)
    demo_regex_cleaning()
    demo_text_normalization()
    build_cleaning_pipeline(stopwords)

    print("\n" + "=" * 60)
    print("停用词过滤与文本清洗 演示完成")
    print("=" * 60)
