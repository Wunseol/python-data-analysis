# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
Pandas读写数据库
- pd.read_sql() 通用读取
- pd.read_sql_query() 执行SQL查询读取
- pd.read_sql_table() 按表名读取 (需要SQLAlchemy)
- df.to_sql() 写入数据库
- if_exists 参数 (fail / replace / append)
- index 参数控制索引写入
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text


def setup_database():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        )
    """)

    products = [
        ("笔记本电脑", "电子产品", 5999.0, 50),
        ("无线鼠标", "电子产品", 89.9, 200),
        ("机械键盘", "电子产品", 299.0, 150),
        ("运动鞋", "服装", 399.0, 80),
        ("羽绒服", "服装", 899.0, 30),
        ("咖啡豆", "食品", 68.0, 500),
        ("巧克力", "食品", 35.0, 300),
    ]
    cursor.executemany(
        "INSERT INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)",
        products,
    )
    conn.commit()
    return conn


def read_sql_query():
    print("=" * 60)
    print("1. pd.read_sql_query() - 执行SQL查询读取")
    print("=" * 60)

    conn = setup_database()

    # 直接传入SQL语句和连接对象
    df = pd.read_sql_query("SELECT * FROM products", conn)
    print("读取全部数据:")
    print(df)
    print(f"\n类型: {type(df)}")
    print(f"形状: {df.shape}")

    # 带条件的查询
    df_electronics = pd.read_sql_query(
        "SELECT * FROM products WHERE category = '电子产品' ORDER BY price DESC",
        conn,
    )
    print("\n电子产品(按价格降序):")
    print(df_electronics)

    # 聚合查询
    df_stats = pd.read_sql_query("""
        SELECT
            category,
            COUNT(*) AS product_count,
            AVG(price) AS avg_price,
            SUM(stock) AS total_stock
        FROM products
        GROUP BY category
    """, conn)
    print("\n品类统计:")
    print(df_stats)

    conn.close()
    print()


def read_sql_with_params():
    print("=" * 60)
    print("2. pd.read_sql_query() 带参数查询")
    print("=" * 60)

    conn = setup_database()

    # 位置参数
    df = pd.read_sql_query(
        "SELECT * FROM products WHERE price > ? AND category = ?",
        conn,
        params=(100.0, "电子产品"),
    )
    print("价格>100的电子产品:")
    print(df)

    # 命名参数 (使用字典)
    df2 = pd.read_sql_query(
        "SELECT * FROM products WHERE category = :cat AND stock > :min_stock",
        conn,
        params={"cat": "食品", "min_stock": 200},
    )
    print("\n库存>200的食品:")
    print(df2)

    conn.close()
    print()


def read_sql_with_sqlalchemy():
    print("=" * 60)
    print("3. 使用 SQLAlchemy引擎 读取数据")
    print("=" * 60)

    # 创建内存数据库引擎
    engine = create_engine("sqlite:///:memory:")

    # 先写入一些数据
    df_sample = pd.DataFrame({
        "name": ["产品A", "产品B", "产品C"],
        "value": [100, 200, 300],
    })
    df_sample.to_sql("items", engine, index=False, if_exists="replace")

    # pd.read_sql() - 通用读取，可传入SQL或表名
    df = pd.read_sql("SELECT * FROM items", engine)
    print("pd.read_sql() 使用引擎:")
    print(df)

    # pd.read_sql_query() - 只接受SQL语句
    df2 = pd.read_sql_query("SELECT * FROM items WHERE value > 150", engine)
    print("\npd.read_sql_query() 条件查询:")
    print(df2)

    # pd.read_sql_table() - 按表名读取 (仅支持SQLAlchemy引擎)
    df3 = pd.read_sql_table("items", engine)
    print("\npd.read_sql_table() 按表名读取:")
    print(df3)

    # 使用 text() 包装SQL (SQLAlchemy 2.0推荐)
    with engine.connect() as connection:
        df4 = pd.read_sql(text("SELECT * FROM items WHERE value > :threshold"), connection, params={"threshold": 100})
    print("\n使用 text() 和命名参数:")
    print(df4)

    print()


def to_sql_basic():
    print("=" * 60)
    print("4. df.to_sql() 写入数据库 - 基本用法")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")

    # 创建DataFrame
    df = pd.DataFrame({
        "name": ["张三", "李四", "王五", "赵六"],
        "age": [25, 30, 28, 35],
        "city": ["北京", "上海", "广州", "深圳"],
        "salary": [15000.0, 18000.0, 12000.0, 20000.0],
    })
    print("原始DataFrame:")
    print(df)

    # 写入数据库
    df.to_sql("employees", conn, index=False, if_exists="replace")
    print("\n已写入数据库表: employees")

    # 验证
    result = pd.read_sql_query("SELECT * FROM employees", conn)
    print("从数据库读回:")
    print(result)

    conn.close()
    print()


def to_sql_if_exists():
    print("=" * 60)
    print("5. if_exists 参数: fail / replace / append")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")

    df1 = pd.DataFrame({"name": ["张三", "李四"], "score": [85, 92]})
    df2 = pd.DataFrame({"name": ["王五", "赵六"], "score": [78, 88]})

    # if_exists='replace': 如果表存在则替换
    df1.to_sql("students", conn, index=False, if_exists="replace")
    print("首次写入 (replace):")
    print(pd.read_sql_query("SELECT * FROM students", conn))

    # if_exists='append': 如果表存在则追加
    df2.to_sql("students", conn, index=False, if_exists="append")
    print("\n追加写入 (append):")
    print(pd.read_sql_query("SELECT * FROM students", conn))

    # if_exists='fail': 如果表存在则报错 (默认值)
    try:
        df1.to_sql("students", conn, index=False, if_exists="fail")
    except ValueError as e:
        print(f"\nif_exists='fail' 报错: {e}")

    conn.close()
    print()


def to_sql_index_param():
    print("=" * 60)
    print("6. index 参数 - 控制DataFrame索引是否写入")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")

    df = pd.DataFrame(
        {"name": ["张三", "李四", "王五"], "score": [85, 92, 78]},
        index=["a", "b", "c"],
    )
    print("带自定义索引的DataFrame:")
    print(df)

    # index=True (默认): 将DataFrame索引作为一列写入
    df.to_sql("students_with_index", conn, index=True, if_exists="replace")
    print("\nindex=True 写入结果:")
    print(pd.read_sql_query("SELECT * FROM students_with_index", conn))

    # index=False: 不写入索引列
    df.to_sql("students_no_index", conn, index=False, if_exists="replace")
    print("\nindex=False 写入结果:")
    print(pd.read_sql_query("SELECT * FROM students_no_index", conn))

    conn.close()
    print()


def to_sql_dtype():
    print("=" * 60)
    print("7. 指定列数据类型 (dtype参数)")
    print("=" * 60)

    from sqlalchemy import Column, Float, Integer, String

    engine = create_engine("sqlite:///:memory:")

    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["产品A", "产品B", "产品C"],
        "price": [99.9, 199.0, 49.5],
        "stock": [100, 200, 50],
    })

    # 使用dtype指定列的数据类型
    df.to_sql(
        "products",
        engine,
        index=False,
        if_exists="replace",
        dtype={
            "id": Integer,
            "name": String(50),
            "price": Float,
            "stock": Integer,
        },
    )

    # 验证表结构
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(products)"))
        print("表结构:")
        for row in result:
            print(f"  {row}")

    # 读回数据
    df_back = pd.read_sql_table("products", engine)
    print("\n读回数据:")
    print(df_back)

    print()


def round_trip_workflow():
    print("=" * 60)
    print("8. 完整工作流: DataFrame → 数据库 → SQL分析 → DataFrame")
    print("=" * 60)

    engine = create_engine("sqlite:///:memory:")

    # 步骤1: 创建DataFrame
    sales_data = pd.DataFrame({
        "product": ["笔记本", "鼠标", "键盘", "笔记本", "鼠标", "键盘", "显示器"],
        "category": ["电子", "电子", "电子", "电子", "电子", "电子", "电子"],
        "amount": [5999, 89, 299, 5999, 89, 299, 1899],
        "quantity": [2, 10, 5, 1, 8, 3, 4],
        "date": ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-02"],
    })
    print("步骤1 - 原始销售数据:")
    print(sales_data)

    # 步骤2: 写入数据库
    sales_data.to_sql("sales", engine, index=False, if_exists="replace")
    print("\n步骤2 - 数据已写入数据库")

    # 步骤3: 用SQL进行分析
    query = text("""
        SELECT
            product AS 产品,
            SUM(quantity) AS 总销量,
            SUM(amount * quantity) AS 总销售额,
            AVG(amount) AS 平均单价
        FROM sales
        GROUP BY product
        ORDER BY 总销售额 DESC
    """)
    df_analysis = pd.read_sql_query(query, engine)
    print("\n步骤3 - SQL分析结果:")
    print(df_analysis)

    # 步骤4: 将分析结果再写回数据库
    df_analysis.to_sql("sales_summary", engine, index=False, if_exists="replace")
    print("\n步骤4 - 分析结果已写回数据库")

    # 步骤5: 验证
    df_final = pd.read_sql_table("sales_summary", engine)
    print("\n步骤5 - 从数据库读回验证:")
    print(df_final)

    print()


if __name__ == "__main__":
    read_sql_query()
    read_sql_with_params()
    read_sql_with_sqlalchemy()
    to_sql_basic()
    to_sql_if_exists()
    to_sql_index_param()
    to_sql_dtype()
    round_trip_workflow()
    print("所有Pandas读写数据库演示完成！")
