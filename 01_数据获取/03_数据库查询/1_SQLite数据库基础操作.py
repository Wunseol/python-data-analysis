# 数据源: SQLite内存数据库 (无需外部数据库配置)
# 依赖库最低版本要求: pandas>=2.0, sqlalchemy>=2.0
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
SQLite数据库基础操作
- sqlite3.connect(':memory:') 内存数据库连接
- cursor 游标对象
- execute 执行SQL语句
- fetchone / fetchall 获取查询结果
- conn.close() 关闭连接
- 上下文管理器 (with语句) 的使用
"""

import sqlite3


def basic_connection():
    print("=" * 60)
    print("1. 基本连接与关闭")
    print("=" * 60)

    # 连接到内存数据库，数据仅在连接存活期间存在
    conn = sqlite3.connect(":memory:")
    print(f"数据库连接对象: {conn}")
    print(f"数据库类型: SQLite 内存数据库")

    # 关闭连接，内存数据全部丢失
    conn.close()
    print("连接已关闭\n")


def cursor_and_execute():
    print("=" * 60)
    print("2. 游标对象与SQL执行")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")

    # 创建游标对象，游标用于执行SQL和获取结果
    cursor = conn.cursor()
    print(f"游标对象: {cursor}")

    # execute() 执行单条SQL语句
    cursor.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    print("已创建表: students")

    # 插入单条数据
    cursor.execute("INSERT INTO students (name, score) VALUES ('张三', 85.5)")
    cursor.execute("INSERT INTO students (name, score) VALUES ('李四', 92.0)")
    # 提交事务，使插入生效
    conn.commit()
    print("已插入2条数据并提交\n")

    conn.close()


def fetch_results():
    print("=" * 60)
    print("3. 获取查询结果: fetchone / fetchall / fetchmany")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 准备数据
    cursor.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    students_data = [
        ("张三", 85.5), ("李四", 92.0), ("王五", 78.3),
        ("赵六", 95.1), ("孙七", 88.7),
    ]
    cursor.executemany("INSERT INTO students (name, score) VALUES (?, ?)", students_data)
    conn.commit()

    # fetchone: 获取一条结果，返回元组或None
    cursor.execute("SELECT * FROM students")
    one_row = cursor.fetchone()
    print(f"fetchone 结果: {one_row}")
    print(f"  类型: {type(one_row)}")
    print(f"  通过索引访问: id={one_row[0]}, name={one_row[1]}, score={one_row[2]}")

    # fetchone 会移动游标，再次调用获取下一条
    next_row = cursor.fetchone()
    print(f"再次 fetchone: {next_row}")

    # fetchall: 获取所有剩余结果，返回列表
    cursor.execute("SELECT * FROM students")
    all_rows = cursor.fetchall()
    print(f"\nfetchall 结果 (共{len(all_rows)}条):")
    for row in all_rows:
        print(f"  {row}")

    # fetchmany: 获取指定数量的结果
    cursor.execute("SELECT * FROM students")
    some_rows = cursor.fetchmany(3)
    print(f"\nfetchmany(3) 结果:")
    for row in some_rows:
        print(f"  {row}")

    # 遍历游标: 也可以直接迭代cursor对象
    print("\n直接遍历游标:")
    cursor.execute("SELECT name, score FROM students")
    for row in cursor:
        print(f"  姓名: {row[0]}, 分数: {row[1]}")

    conn.close()
    print()


def row_factory():
    print("=" * 60)
    print("4. 使用 row_factory 让结果以字典形式返回")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")

    # 设置 row_factory 为 sqlite3.Row，可以通过列名访问
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    cursor.executemany(
        "INSERT INTO students (name, score) VALUES (?, ?)",
        [("张三", 85.5), ("李四", 92.0)],
    )
    conn.commit()

    cursor.execute("SELECT * FROM students")
    for row in cursor:
        print(f"  dict(row): {dict(row)}")
        print(f"  通过列名访问: id={row['id']}, name={row['name']}, score={row['score']}")

    conn.close()
    print()


def context_manager():
    print("=" * 60)
    print("5. 上下文管理器 (with语句) 的使用")
    print("=" * 60)

    # with conn 管理事务: 成功自动commit，异常自动rollback
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, score REAL)")

    # 使用 with conn 管理事务
    with conn:
        conn.execute("INSERT INTO students (name, score) VALUES ('张三', 85.5)")
        conn.execute("INSERT INTO students (name, score) VALUES ('李四', 92.0)")
    # 离开with块时自动commit

    # 验证数据已提交
    cursor.execute("SELECT * FROM students")
    print("事务提交后的数据:")
    for row in cursor.fetchall():
        print(f"  {row}")

    # 模拟事务回滚
    try:
        with conn:
            conn.execute("INSERT INTO students (name, score) VALUES ('王五', 78.3)")
            # 故意引发异常
            raise ValueError("模拟异常")
    except ValueError as e:
        print(f"\n捕获异常: {e}")
        print("事务已自动回滚，王五的数据不会被插入")

    # 验证回滚结果
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]
    print(f"当前表中记录数: {count}")

    conn.close()
    print()


def connection_properties():
    print("=" * 60)
    print("6. 连接对象常用属性与方法")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")

    # lastrowid: 获取最后插入行的rowid
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("INSERT INTO items (name) VALUES ('物品A')")
    print(f"最后插入行的ID: {cursor.lastrowid}")

    # rowcount: 获取受影响的行数
    cursor.execute("INSERT INTO items (name) VALUES ('物品B')")
    cursor.execute("INSERT INTO items (name) VALUES ('物品C')")
    cursor.execute("UPDATE items SET name = '物品A-修改' WHERE name = '物品A'")
    print(f"UPDATE 受影响行数: {cursor.rowcount}")

    # total_changes: 自连接以来的总修改行数
    print(f"总修改行数: {conn.total_changes}")

    # isolation_level: 事务隔离级别
    print(f"隔离级别: {conn.isolation_level}")

    # in_transaction: 是否在事务中
    print(f"是否在事务中: {conn.in_transaction}")

    conn.close()
    print()


if __name__ == "__main__":
    basic_connection()
    cursor_and_execute()
    fetch_results()
    row_factory()
    context_manager()
    connection_properties()
    print("所有SQLite基础操作演示完成！")
