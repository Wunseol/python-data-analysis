# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
查询与条件筛选
- SELECT 基本查询
- WHERE 条件筛选
- ORDER BY 排序
- LIMIT 限制行数
- LIKE 模糊匹配
- IN 集合筛选
- BETWEEN 范围筛选
- DISTINCT 去重
- 别名 (AS)
"""

import sqlite3


def setup_database():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            age INTEGER,
            city TEXT
        )
    """)

    employees = [
        ("张三", "技术部", 15000.0, 28, "北京"),
        ("李四", "技术部", 18000.0, 32, "上海"),
        ("王五", "市场部", 12000.0, 25, "广州"),
        ("赵六", "市场部", 13500.0, 30, "深圳"),
        ("孙七", "财务部", 14000.0, 35, "北京"),
        ("周八", "财务部", 16000.0, 40, "上海"),
        ("吴九", "技术部", 20000.0, 38, "杭州"),
        ("郑十", "人事部", 11000.0, 26, "成都"),
        ("钱十一", "人事部", 12500.0, 29, "武汉"),
        ("陈十二", "技术部", 17000.0, 34, "北京"),
        ("林十三", "市场部", 11500.0, 27, "广州"),
        ("黄十四", "财务部", 15500.0, 42, "深圳"),
    ]
    cursor.executemany(
        "INSERT INTO employees (name, department, salary, age, city) VALUES (?, ?, ?, ?, ?)",
        employees,
    )
    conn.commit()
    return conn


def select_basic():
    print("=" * 60)
    print("1. SELECT 基本查询")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 查询所有列
    cursor.execute("SELECT * FROM employees")
    print("SELECT * (所有列):")
    for row in cursor.fetchall()[:3]:
        print(f"  {dict(row)}")

    # 查询指定列
    cursor.execute("SELECT name, department, salary FROM employees")
    print("\nSELECT 指定列:")
    for row in cursor.fetchall()[:3]:
        print(f"  {dict(row)}")

    conn.close()
    print()


def where_conditions():
    print("=" * 60)
    print("2. WHERE 条件筛选")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 等值条件
    cursor.execute("SELECT name, salary FROM employees WHERE department = '技术部'")
    print("技术部员工:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    # 比较条件
    cursor.execute("SELECT name, salary FROM employees WHERE salary > 15000")
    print("\n薪资 > 15000:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    # AND / OR 组合条件
    cursor.execute(
        "SELECT name, department, salary FROM employees WHERE department = '技术部' AND salary > 16000"
    )
    print("\n技术部且薪资>16000:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['department']}, {row['salary']}")

    cursor.execute(
        "SELECT name, department, city FROM employees WHERE city = '北京' OR city = '上海'"
    )
    print("\n北京或上海:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['city']}")

    conn.close()
    print()


def order_by():
    print("=" * 60)
    print("3. ORDER BY 排序")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 升序 (ASC, 默认)
    cursor.execute("SELECT name, salary FROM employees ORDER BY salary ASC")
    print("薪资升序:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    # 降序 (DESC)
    cursor.execute("SELECT name, salary FROM employees ORDER BY salary DESC")
    print("\n薪资降序:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    # 多列排序: 先按部门升序，再按薪资降序
    cursor.execute("SELECT name, department, salary FROM employees ORDER BY department ASC, salary DESC")
    print("\n部门升序 + 薪资降序:")
    for row in cursor.fetchall():
        print(f"  {row['department']:<6} {row['name']:<6} {row['salary']}")

    conn.close()
    print()


def limit_query():
    print("=" * 60)
    print("4. LIMIT 限制行数")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # LIMIT: 限制返回行数
    cursor.execute("SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3")
    print("薪资最高的3人:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    # LIMIT + OFFSET: 分页查询
    page_size = 3
    page = 2
    offset = (page - 1) * page_size
    cursor.execute("SELECT name, salary FROM employees ORDER BY id LIMIT ? OFFSET ?", (page_size, offset))
    print(f"\n第{page}页数据 (每页{page_size}条):")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    conn.close()
    print()


def like_pattern():
    print("=" * 60)
    print("5. LIKE 模糊匹配")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # % 匹配任意0个或多个字符
    cursor.execute("SELECT name FROM employees WHERE name LIKE '张%'")
    print("姓'张'的员工:")
    for row in cursor.fetchall():
        print(f"  {row['name']}")

    # _ 匹配单个字符
    cursor.execute("SELECT name FROM employees WHERE name LIKE '张_'")
    print("\n名字为'张X'(两字)的员工:")
    for row in cursor.fetchall():
        print(f"  {row['name']}")

    # 包含匹配
    cursor.execute("SELECT name, city FROM employees WHERE city LIKE '%京%'")
    print("\n城市包含'京'的员工:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['city']}")

    # NOT LIKE
    cursor.execute("SELECT name, city FROM employees WHERE city NOT LIKE '%京%'")
    print("\n城市不包含'京'的员工:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['city']}")

    conn.close()
    print()


def in_between():
    print("=" * 60)
    print("6. IN 与 BETWEEN 筛选")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # IN: 在指定集合中
    cursor.execute("SELECT name, department FROM employees WHERE department IN ('技术部', '财务部')")
    print("技术部或财务部:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['department']}")

    # NOT IN
    cursor.execute("SELECT name, department FROM employees WHERE department NOT IN ('技术部', '财务部')")
    print("\n非技术部且非财务部:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['department']}")

    # BETWEEN: 范围筛选 (包含边界)
    cursor.execute("SELECT name, salary FROM employees WHERE salary BETWEEN 13000 AND 16000")
    print("\n薪资 13000~16000:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['salary']}")

    # BETWEEN 也适用于日期和文本
    cursor.execute("SELECT name, age FROM employees WHERE age BETWEEN 30 AND 40")
    print("\n年龄 30~40:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['age']}")

    conn.close()
    print()


def distinct_and_alias():
    print("=" * 60)
    print("7. DISTINCT 去重与 别名(AS)")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # DISTINCT: 去重
    cursor.execute("SELECT DISTINCT department FROM employees")
    print("所有部门(去重):")
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    cursor.execute("SELECT DISTINCT city FROM employees ORDER BY city")
    print("\n所有城市(去重):")
    for row in cursor.fetchall():
        print(f"  {row[0]}")

    # 别名: AS (AS关键字可省略)
    cursor.execute("""
        SELECT
            name AS 姓名,
            department AS 部门,
            salary AS 月薪,
            salary * 12 AS 年薪
        FROM employees
        WHERE department = '技术部'
    """)
    print("\n使用别名:")
    print(f"  {'姓名':<6} {'部门':<6} {'月薪':<10} {'年薪':<10}")
    for row in cursor.fetchall():
        print(f"  {row['姓名']:<6} {row['部门']:<6} {row['月薪']:<10} {row['年薪']:<10}")

    # 表别名
    cursor.execute("""
        SELECT e.name, e.department, e.salary
        FROM employees AS e
        WHERE e.salary > 15000
    """)
    print("\n表别名查询 (薪资>15000):")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[2]}")

    conn.close()
    print()


if __name__ == "__main__":
    select_basic()
    where_conditions()
    order_by()
    limit_query()
    like_pattern()
    in_between()
    distinct_and_alias()
    print("所有查询与条件筛选演示完成！")
