# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
创建表与插入数据
- CREATE TABLE 与数据类型
- INSERT INTO 插入数据
- executemany 批量插入
- INSERT OR REPLACE 替换插入
- 自增主键 (AUTOINCREMENT)
"""

import sqlite3


def create_table_with_types():
    print("=" * 60)
    print("1. CREATE TABLE 与 SQLite 数据类型")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # SQLite 数据类型: TEXT(文本), INTEGER(整数), REAL(浮点数), BLOB(二进制), NULL(空值)
    # SQLite 采用动态类型系统，列的类型更多是建议而非强制
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0,
            description TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)
    print("已创建表: products")
    print("  字段: id(INTEGER主键自增), name(TEXT非空), price(REAL非空)")
    print("        stock(INTEGER默认0), description(TEXT), is_active(INTEGER默认1)")

    # 查看表结构
    cursor.execute("PRAGMA table_info(products)")
    print("\n表结构信息 (PRAGMA table_info):")
    print(f"  {'序号':<4} {'字段名':<14} {'类型':<10} {'非空':<4} {'默认值':<8} {'主键':<4}")
    for col in cursor.fetchall():
        print(f"  {col[0]:<4} {col[1]:<14} {col[2]:<10} {col[3]:<4} {str(col[4]):<8} {col[5]:<4}")

    conn.close()
    print()


def insert_single():
    print("=" * 60)
    print("2. INSERT INTO 插入单条数据")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        )
    """)

    # 方式1: 位置参数 (推荐，防止SQL注入)
    cursor.execute(
        "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
        ("笔记本电脑", 5999.0, 50),
    )
    print(f"插入第1条, lastrowid={cursor.lastrowid}")

    # 方式2: 命名参数
    cursor.execute(
        "INSERT INTO products (name, price, stock) VALUES (:name, :price, :stock)",
        {"name": "无线鼠标", "price": 89.9, "stock": 200},
    )
    print(f"插入第2条, lastrowid={cursor.lastrowid}")

    # 方式3: 省略列名(不推荐，依赖列的顺序)
    # cursor.execute("INSERT INTO products VALUES (NULL, '键盘', 299.0, 150)")

    conn.commit()

    # 查看结果
    cursor.execute("SELECT * FROM products")
    print("\n当前数据:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


def insert_batch():
    print("=" * 60)
    print("3. executemany 批量插入")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        )
    """)

    # 准备批量数据
    products = [
        ("笔记本电脑", 5999.0, 50),
        ("无线鼠标", 89.9, 200),
        ("机械键盘", 299.0, 150),
        ("显示器", 1899.0, 30),
        ("USB集线器", 49.9, 500),
        ("耳机", 199.0, 100),
        ("摄像头", 159.0, 80),
        ("移动硬盘", 399.0, 60),
    ]

    # executemany: 批量执行同一条SQL，参数不同
    cursor.executemany(
        "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
        products,
    )
    conn.commit()
    print(f"批量插入 {cursor.rowcount} 条数据")

    # 查看结果
    cursor.execute("SELECT * FROM products")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


def insert_or_replace():
    print("=" * 60)
    print("4. INSERT OR REPLACE 替换插入")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL
        )
    """)

    # 先插入一条数据
    cursor.execute("INSERT INTO products (id, name, price) VALUES (1, '笔记本电脑', 5999.0)")
    conn.commit()

    cursor.execute("SELECT * FROM products")
    print(f"原始数据: {cursor.fetchall()}")

    # INSERT OR REPLACE: 如果主键冲突则替换整行
    cursor.execute("INSERT OR REPLACE INTO products (id, name, price) VALUES (1, '笔记本电脑Pro', 7999.0)")
    conn.commit()

    cursor.execute("SELECT * FROM products")
    print(f"REPLACE后: {cursor.fetchall()}")

    # INSERT OR IGNORE: 如果主键冲突则忽略
    cursor.execute("INSERT OR IGNORE INTO products (id, name, price) VALUES (1, '笔记本电脑Max', 9999.0)")
    conn.commit()

    cursor.execute("SELECT * FROM products")
    print(f"IGNORE后(无变化): {cursor.fetchall()}")

    # 对比: 普通INSERT主键冲突会报错
    try:
        cursor.execute("INSERT INTO products (id, name, price) VALUES (1, '冲突产品', 0)")
    except sqlite3.IntegrityError as e:
        print(f"普通INSERT主键冲突报错: {e}")

    conn.close()
    print()


def auto_increment():
    print("=" * 60)
    print("5. 自增主键 AUTOINCREMENT")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # INTEGER PRIMARY KEY 自动自增 (不加AUTOINCREMENT关键字也可以)
    # AUTOINCREMENT 关键字保证ID永不复用(即使删除了行)
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT
        )
    """)

    # 插入数据，不指定id
    cursor.executemany(
        "INSERT INTO users (username, email) VALUES (?, ?)",
        [
            ("alice", "alice@example.com"),
            ("bob", "bob@example.com"),
            ("charlie", "charlie@example.com"),
        ],
    )
    conn.commit()

    cursor.execute("SELECT * FROM users")
    print("插入后的数据:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # 删除id=2的记录
    cursor.execute("DELETE FROM users WHERE id = 2")
    conn.commit()
    print("\n删除id=2后:")

    cursor.execute("SELECT * FROM users")
    for row in cursor.fetchall():
        print(f"  {row}")

    # 再插入新记录，id不会复用2，而是从4开始
    cursor.execute("INSERT INTO users (username, email) VALUES ('david', 'david@example.com')")
    conn.commit()

    cursor.execute("SELECT * FROM users")
    print("\n再插入后(id不复用):")
    for row in cursor.fetchall():
        print(f"  {row}")

    # 查看自增序列
    cursor.execute("SELECT * FROM sqlite_sequence")
    print(f"\nsqlite_sequence 表内容: {cursor.fetchall()}")

    conn.close()
    print()


def create_table_constraints():
    print("=" * 60)
    print("6. 表约束: NOT NULL, UNIQUE, DEFAULT, CHECK")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_no TEXT NOT NULL UNIQUE,
            customer_name TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount > 0),
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'paid', 'shipped', 'completed')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("已创建带约束的表: orders")

    # 正常插入
    cursor.execute(
        "INSERT INTO orders (order_no, customer_name, amount) VALUES (?, ?, ?)",
        ("ORD001", "张三", 199.0),
    )
    print("正常插入成功")

    # 违反UNIQUE约束
    try:
        cursor.execute(
            "INSERT INTO orders (order_no, customer_name, amount) VALUES (?, ?, ?)",
            ("ORD001", "李四", 299.0),
        )
    except sqlite3.IntegrityError as e:
        print(f"违反UNIQUE约束: {e}")

    # 违反CHECK约束 (amount <= 0)
    try:
        cursor.execute(
            "INSERT INTO orders (order_no, customer_name, amount) VALUES (?, ?, ?)",
            ("ORD002", "王五", -10.0),
        )
    except sqlite3.IntegrityError as e:
        print(f"违反CHECK约束: {e}")

    # 使用DEFAULT值
    cursor.execute(
        "INSERT INTO orders (order_no, customer_name, amount, status) VALUES (?, ?, ?, ?)",
        ("ORD002", "王五", 599.0, "paid"),
    )
    conn.commit()

    cursor.execute("SELECT * FROM orders")
    print("\n当前数据:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


if __name__ == "__main__":
    create_table_with_types()
    insert_single()
    insert_batch()
    insert_or_replace()
    auto_increment()
    create_table_constraints()
    print("所有创建表与插入数据演示完成！")
