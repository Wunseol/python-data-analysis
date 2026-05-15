# -*- coding: utf-8 -*-
"""
文本读取与编码处理

数据来源: 脚本内自建文本数据
依赖库最低版本要求: jieba>=0.42, wordcloud>=1.9, scikit-learn>=1.3, pandas>=2.0
# [注意] jieba 长期未更新但功能稳定，如需更活跃维护的替代方案可考虑 pkuseg
"""

import os
import tempfile
from pathlib import Path
from collections import Counter


def demo_basic_read_write():
    print("=" * 60)
    print("1. 基本文本读写操作")
    print("=" * 60)

    sample_text = "这是第一行文本。\n这是第二行文本。\n这是第三行文本。"

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "sample_utf8.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(f"文件已写入: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"\nread() 读取全部内容:\n{content}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"\nreadlines() 按行读取 (共 {len(lines)} 行):")
    for i, line in enumerate(lines, 1):
        print(f"  第{i}行: {line.strip()}")

    with open(file_path, "r", encoding="utf-8") as f:
        print("\n逐行迭代读取:")
        for i, line in enumerate(f, 1):
            print(f"  第{i}行: {line.strip()}")

    return tmp_dir


def demo_encoding_handling(tmp_dir):
    print("\n" + "=" * 60)
    print("2. 编码处理: UTF-8 / GBK / GB2312")
    print("=" * 60)

    chinese_text = "数据分析是当今最重要的技能之一，文本数据分析更是不可或缺。"

    utf8_path = os.path.join(tmp_dir, "sample_utf8.txt")
    gbk_path = os.path.join(tmp_dir, "sample_gbk.txt")
    gb2312_path = os.path.join(tmp_dir, "sample_gb2312.txt")

    with open(utf8_path, "w", encoding="utf-8") as f:
        f.write(chinese_text)
    with open(gbk_path, "w", encoding="gbk") as f:
        f.write(chinese_text)
    with open(gb2312_path, "w", encoding="gb2312") as f:
        f.write(chinese_text)

    for path, enc in [(utf8_path, "utf-8"), (gbk_path, "gbk"), (gb2312_path, "gb2312")]:
        with open(path, "r", encoding=enc) as f:
            content = f.read()
        print(f"  {enc:>8} 编码读取: {content}")

    print("\n  用错误编码读取会抛出异常:")
    try:
        with open(gbk_path, "r", encoding="utf-8") as f:
            f.read()
    except UnicodeDecodeError as e:
        print(f"  UnicodeDecodeError: {e}")

    print("\n  使用 errors 参数处理编码错误:")
    with open(gbk_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    print(f"  errors='replace' 读取结果: {content}")


def demo_chardet():
    print("\n" + "=" * 60)
    print("3. 使用 chardet 检测文件编码")
    print("=" * 60)

    try:
        import chardet
    except ImportError:
        print("  chardet 未安装，请运行: pip install chardet")
        print("  以下为概念演示:")
        print("""
  import chardet

  with open("unknown_encoding.txt", "rb") as f:
      raw_data = f.read()
  result = chardet.detect(raw_data)
  print(result)
  # 输出示例: {'encoding': 'GB2312', 'confidence': 0.99, 'language': 'Chinese'}

  detected_encoding = result['encoding']
  with open("unknown_encoding.txt", "r", encoding=detected_encoding) as f:
      text = f.read()
        """)
        return

    tmp_dir = tempfile.mkdtemp()
    gbk_path = os.path.join(tmp_dir, "test_gbk.txt")
    with open(gbk_path, "w", encoding="gbk") as f:
        f.write("这是一段用于测试编码检测的中文文本。")

    with open(gbk_path, "rb") as f:
        raw_data = f.read()

    result = chardet.detect(raw_data)
    print(f"  检测结果: {result}")
    print(f"  检测到的编码: {result['encoding']}")
    print(f"  置信度: {result['confidence']:.2%}")

    detected_encoding = result['encoding']
    with open(gbk_path, "r", encoding=detected_encoding) as f:
        content = f.read()
    print(f"  使用检测到的编码读取: {content}")


def demo_pathlib():
    print("\n" + "=" * 60)
    print("4. 使用 pathlib 处理文件路径")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp())

    file_path = tmp_dir / "data" / "text_analysis" / "sample.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text("使用 pathlib 写入的文本内容", encoding="utf-8")
    print(f"  文件路径: {file_path}")
    print(f"  父目录: {file_path.parent}")
    print(f"  文件名: {file_path.name}")
    print(f"  文件后缀: {file_path.suffix}")
    print(f"  是否存在: {file_path.exists()}")

    content = file_path.read_text(encoding="utf-8")
    print(f"  读取内容: {content}")

    txt_files = list(tmp_dir.rglob("*.txt"))
    print(f"\n  递归查找所有 .txt 文件:")
    for f in txt_files:
        print(f"    {f}")

    print("\n  Path 对象 vs os.path 对比:")
    print(f"    pathlib: {file_path.parent / 'new_file.txt'}")
    print(f"    os.path: {os.path.join(os.path.dirname(str(file_path)), 'new_file.txt')}")


def demo_write_modes():
    print("\n" + "=" * 60)
    print("5. 文件写入模式与最佳实践")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "modes_demo.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("第一行: 写模式 (w) 会覆盖已有内容\n")
    print("  模式 'w': 覆盖写入")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("第二行: 追加模式 (a) 在末尾添加\n")
    print("  模式 'a': 追加写入")

    with open(file_path, "r", encoding="utf-8") as f:
        print(f"\n  当前文件内容:\n  {f.read().replace(chr(10), chr(10) + '  ')}")

    lines = ["第三行: 写入列表\n", "第四行: writelines 方法\n"]
    with open(file_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print("  writelines(): 写入字符串列表")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"\n  最终文件内容:\n  {content.replace(chr(10), chr(10) + '  ')}")

    print("\n  编码最佳实践:")
    print("    1. 始终显式指定 encoding 参数，不要依赖系统默认编码")
    print("    2. 推荐使用 utf-8 作为默认编码")
    print("    3. 处理中文文本时注意 GBK/GB2312 编码的文件")
    print("    4. 读取未知编码文件时使用 chardet 检测")
    print("    5. 使用 with 语句确保文件正确关闭")


def demo_batch_read():
    print("\n" + "=" * 60)
    print("6. 批量读取文本文件")
    print("=" * 60)

    tmp_dir = Path(tempfile.mkdtemp())

    docs = {
        "tech.txt": "人工智能技术正在改变世界，深度学习模型越来越强大。",
        "finance.txt": "股票市场今日波动较大，投资者需谨慎操作。",
        "sports.txt": "中国女排在比赛中展现了顽强的拼搏精神。",
    }

    for name, text in docs.items():
        (tmp_dir / name).write_text(text, encoding="utf-8")

    print(f"  目录: {tmp_dir}")
    print(f"  文件列表: {list(tmp_dir.glob('*.txt'))}")

    corpus = {}
    for f in tmp_dir.glob("*.txt"):
        corpus[f.stem] = f.read_text(encoding="utf-8")

    print("\n  批量读取结果:")
    for name, text in corpus.items():
        print(f"    {name}: {text}")

    print(f"\n  共读取 {len(corpus)} 个文件")


if __name__ == "__main__":
    tmp_dir = demo_basic_read_write()
    demo_encoding_handling(tmp_dir)
    demo_chardet()
    demo_pathlib()
    demo_write_modes()
    demo_batch_read()

    print("\n" + "=" * 60)
    print("文本读取与编码处理 演示完成")
    print("=" * 60)
