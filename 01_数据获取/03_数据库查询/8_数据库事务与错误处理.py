# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
数据库事务与错误处理
- conn.commit() 提交事务
- conn.rollback() 回滚事务
- try/except/finally 模式
- IntegrityError 完整性错误
- OperationalError 操作错误
- 事务隔离概念
"""

import sqlite3
from sqlalchemy import create_engine, String, Integer, Float, Column, text
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)


def commit_and_rollback():
    print("=" * 60)
    print("1. commit() 与 rollback() 基本概念")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            balance REAL NOT NULL CHECK(balance >= 0)
        )
    """)
    cursor.executemany(
        "INSERT INTO accounts (name, balance) VALUES (?, ?)",
        [("张三", 10000.0), ("李四", 5000.0)],
    )
    conn.commit()

    # 查看初始余额
    cursor.execute("SELECT name, balance FROM accounts")
    print("初始余额:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    # 模拟转账: 张三 -> 李四 转账2000
    print("\n--- 转账: 张三 -> 李四 2000元 ---")
    cursor.execute("UPDATE accounts SET balance = balance - 2000 WHERE name = '张三'")
    cursor.execute("UPDATE accounts SET balance = balance + 2000 WHERE name = '李四'")
    conn.commit()

    cursor.execute("SELECT name, balance FROM accounts")
    print("转账后余额:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    # 模拟回滚: 张三 -> 李四 转账50000 (余额不足)
    print("\n--- 尝试转账: 张三 -> 李四 50000元 (余额不足) ---")
    cursor.execute("UPDATE accounts SET balance = balance - 50000 WHERE name = '张三'")
    # 检查余额是否为负
    cursor.execute("SELECT balance FROM accounts WHERE name = '张三'")
    balance = cursor.fetchone()[0]
    if balance < 0:
        print(f"张三余额为负 ({balance})，执行回滚!")
        conn.rollback()
    else:
        conn.commit()

    cursor.execute("SELECT name, balance FROM accounts")
    print("回滚后余额 (与转账前一致):")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print()


def try_except_finally():
    print("=" * 60)
    print("2. try/except/finally 标准模式")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            price REAL NOT NULL CHECK(price > 0)
        )
    """)
    conn.commit()

    # 标准事务模式
    try:
        cursor.execute("INSERT INTO products (name, price) VALUES ('笔记本电脑', 5999.0)")
        cursor.execute("INSERT INTO products (name, price) VALUES ('无线鼠标', 89.9)")
        conn.commit()
        print("事务1: 正常提交成功")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"事务1: 回滚 - {e}")
    finally:
        pass

    # 事务失败示例
    try:
        cursor.execute("INSERT INTO products (name, price) VALUES ('机械键盘', 299.0)")
        # 故意插入重复名称，违反UNIQUE约束
        cursor.execute("INSERT INTO products (name, price) VALUES ('笔记本电脑', 7999.0)")
        conn.commit()
        print("事务2: 提交成功")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"事务2: 回滚 - {e}")
        print("  (机械键盘的插入也被回滚，整个事务原子性)")

    # 验证: 只有事务1的两条记录
    cursor.execute("SELECT * FROM products")
    print("\n当前数据:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


def integrity_error():
    print("=" * 60)
    print("3. IntegrityError 完整性错误")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            age INTEGER CHECK(age >= 18)
        )
    """)
    conn.commit()

    cursor.execute("INSERT INTO users (username, email, age) VALUES ('alice', 'alice@test.com', 25)")
    conn.commit()

    # 错误1: UNIQUE约束冲突
    try:
        cursor.execute("INSERT INTO users (username, email, age) VALUES ('alice', 'alice2@test.com', 30)")
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"UNIQUE冲突: {e}")

    # 错误2: NOT NULL约束
    try:
        cursor.execute("INSERT INTO users (username, email, age) VALUES (NULL, 'bob@test.com', 28)")
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"NOT NULL冲突: {e}")

    # 错误3: CHECK约束
    try:
        cursor.execute("INSERT INTO users (username, email, age) VALUES ('bob', 'bob@test.com', 15)")
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"CHECK冲突: {e}")

    # 正确插入
    try:
        cursor.execute("INSERT INTO users (username, email, age) VALUES ('bob', 'bob@test.com', 22)")
        conn.commit()
        print("正确插入: bob")
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"插入失败: {e}")

    cursor.execute("SELECT * FROM users")
    print("\n当前用户:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


def operational_error():
    print("=" * 60)
    print("4. OperationalError 操作错误")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 错误1: 表不存在
    try:
        cursor.execute("SELECT * FROM nonexistent_table")
    except sqlite3.OperationalError as e:
        print(f"表不存在: {e}")

    # 错误2: 列不存在
    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    try:
        cursor.execute("SELECT nonexistent_column FROM test")
    except sqlite3.OperationalError as e:
        print(f"列不存在: {e}")

    # 错误3: SQL语法错误
    try:
        cursor.execute("SELCT * FROM test")
    except sqlite3.OperationalError as e:
        print(f"SQL语法错误: {e}")

    # 错误4: 重复创建表
    try:
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    except sqlite3.OperationalError as e:
        print(f"表已存在: {e}")

    conn.close()
    print()


def sqlalchemy_transaction():
    print("=" * 60)
    print("5. SQLAlchemy 2.0 事务管理")
    print("=" * 60)

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    # 方式1: Session 的自动事务管理
    with Session(engine) as session:
        product1 = Product(name="笔记本电脑", price=5999.0, stock=50)
        product2 = Product(name="无线鼠标", price=89.9, stock=200)
        session.add_all([product1, product2])
        session.commit()
        print("Session方式: 成功提交2个产品")

    # 方式2: 使用 engine.begin() 上下文管理器
    # 成功自动commit，异常自动rollback
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO products (name, price, stock) VALUES (:name, :price, :stock)"),
                {"name": "机械键盘", "price": 299.0, "stock": 150},
            )
            print("engine.begin()方式: 成功提交1个产品")
    except Exception as e:
        print(f"事务失败(自动回滚): {e}")

    # 方式3: 事务失败自动回滚
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO products (name, price, stock) VALUES (:name, :price, :stock)"),
                {"name": "显示器", "price": 1899.0, "stock": 30},
            )
            # 故意引发异常
            raise RuntimeError("模拟业务异常")
    except RuntimeError as e:
        print(f"捕获异常: {e}")
        print("  显示器的插入已自动回滚")

    # 验证结果
    with Session(engine) as session:
        result = session.execute(select(Product))
        print("\n当前产品列表:")
        for product in result.scalars():
            print(f"  {product.name}: ¥{product.price}")

    print()


def transaction_isolation_concept():
    print("=" * 60)
    print("6. 事务隔离级别概念")
    print("=" * 60)

    # SQLite 默认隔离级别为 SERIALIZABLE (最高级别)
    # SQLite 支持的隔离级别:
    #   - DEFERRED  (默认): 延迟获取锁，直到第一次读写操作
    #   - IMMEDIATE: 立即获取保留锁
    #   - EXCLUSIVE: 立即获取排他锁

    conn = sqlite3.connect(":memory:", isolation_level="DEFERRED")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE counter (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value INTEGER DEFAULT 0
        )
    """)
    cursor.execute("INSERT INTO counter (name, value) VALUES ('点击量', 0)")
    conn.commit()

    # 演示并发问题的概念 (单线程模拟)
    print("事务隔离级别说明:")
    print("  DEFERRED  - 延迟加锁，直到首次操作 (SQLite默认)")
    print("  IMMEDIATE - 开始事务时立即获取保留锁")
    print("  EXCLUSIVE - 开始事务时立即获取排他锁")
    print()
    print("常见并发问题:")
    print("  脏读 (Dirty Read)       - 读取未提交的数据")
    print("  不可重复读 (Non-repeatable) - 同一查询两次结果不同")
    print("  幻读 (Phantom Read)      - 同一查询两次行数不同")
    print()
    print("SQLite默认SERIALIZABLE隔离级别，可以避免以上所有问题")
    print("但SQLite是文件级锁，写操作会锁定整个数据库")

    # 演示: 使用BEGIN事务
    cursor.execute("BEGIN")
    cursor.execute("UPDATE counter SET value = value + 1 WHERE name = '点击量'")
    cursor.execute("SELECT value FROM counter WHERE name = '点击量'")
    print(f"\n事务中: 点击量 = {cursor.fetchone()[0]}")

    cursor.execute("COMMIT")
    cursor.execute("SELECT value FROM counter WHERE name = '点击量'")
    print(f"提交后: 点击量 = {cursor.fetchone()[0]}")

    conn.close()
    print()


def safe_database_operation():
    print("=" * 60)
    print("7. 安全的数据库操作封装")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            price REAL NOT NULL CHECK(price > 0),
            stock INTEGER DEFAULT 0
        )
    """)
    conn.commit()

    def safe_insert(product_name, product_price, product_stock=0):
        try:
            cursor.execute(
                "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
                (product_name, product_price, product_stock),
            )
            conn.commit()
            return True, f"成功插入: {product_name}"
        except sqlite3.IntegrityError as e:
            conn.rollback()
            return False, f"完整性错误: {e}"
        except sqlite3.OperationalError as e:
            conn.rollback()
            return False, f"操作错误: {e}"
        except Exception as e:
            conn.rollback()
            return False, f"未知错误: {e}"

    results = [
        safe_insert("笔记本电脑", 5999.0, 50),
        safe_insert("无线鼠标", 89.9, 200),
        safe_insert("笔记本电脑", 7999.0, 30),
        safe_insert("键盘", -100.0, 10),
        safe_insert("机械键盘", 299.0, 150),
    ]

    print("安全插入结果:")
    for success, msg in results:
        status = "✓" if success else "✗"
        print(f"  [{status}] {msg}")

    cursor.execute("SELECT * FROM products")
    print("\n最终数据:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()
    print()


if __name__ == "__main__":
    commit_and_rollback()
    try_except_finally()
    integrity_error()
    operational_error()
    sqlalchemy_transaction()
    transaction_isolation_concept()
    safe_database_operation()
    print("所有数据库事务与错误处理演示完成！")
