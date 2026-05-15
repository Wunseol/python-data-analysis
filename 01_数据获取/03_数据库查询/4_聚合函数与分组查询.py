# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
聚合函数与分组查询
- COUNT 计数
- SUM 求和
- AVG 平均值
- MAX / MIN 最大/最小值
- GROUP BY 分组
- HAVING 分组后筛选
- 聚合函数与 NULL 值的处理
"""

import sqlite3


def setup_database():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            quantity INTEGER NOT NULL,
            region TEXT,
            sale_date TEXT
        )
    """)

    sales_data = [
        ("笔记本电脑", "电子产品", 5999.0, 2, "华东", "2024-01-15"),
        ("无线鼠标", "电子产品", 89.9, 10, "华北", "2024-01-18"),
        ("机械键盘", "电子产品", 299.0, 5, "华东", "2024-02-03"),
        ("运动鞋", "服装", 399.0, 8, "华南", "2024-02-10"),
        ("羽绒服", "服装", 899.0, 3, "华北", "2024-02-15"),
        ("咖啡豆", "食品", 68.0, 20, "华东", "2024-03-01"),
        ("巧克力", "食品", 35.0, 15, "华南", "2024-03-05"),
        ("笔记本电脑", "电子产品", 5999.0, 1, "华南", "2024-03-10"),
        ("运动鞋", "服装", 399.0, 5, "华东", "2024-03-15"),
        ("机械键盘", "电子产品", 299.0, 3, "华北", "2024-03-20"),
        ("咖啡豆", "食品", 68.0, 25, "华北", "2024-04-01"),
        ("羽绒服", "服装", 899.0, 2, "华南", "2024-04-05"),
        ("无线鼠标", "电子产品", 89.9, 8, "华东", "2024-04-10"),
        ("笔记本电脑", "电子产品", 5999.0, 3, "华北", "2024-04-15"),
        ("巧克力", "食品", 35.0, 30, "华东", "2024-04-20"),
    ]
    cursor.executemany(
        "INSERT INTO sales (product, category, amount, quantity, region, sale_date) VALUES (?, ?, ?, ?, ?, ?)",
        sales_data,
    )
    conn.commit()
    return conn


def count_function():
    print("=" * 60)
    print("1. COUNT 计数函数")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # COUNT(*) 计算所有行数
    cursor.execute("SELECT COUNT(*) AS total FROM sales")
    print(f"总记录数: {cursor.fetchone()['total']}")

    # COUNT(column) 计算非NULL值的数量
    cursor.execute("SELECT COUNT(region) AS non_null_count FROM sales")
    print(f"region非NULL的记录数: {cursor.fetchone()['non_null_count']}")

    # COUNT(DISTINCT column) 计算不重复值的数量
    cursor.execute("SELECT COUNT(DISTINCT product) AS unique_products FROM sales")
    print(f"不同产品数: {cursor.fetchone()['unique_products']}")

    cursor.execute("SELECT COUNT(DISTINCT category) AS unique_categories FROM sales")
    print(f"不同品类数: {cursor.fetchone()['unique_categories']}")

    conn.close()
    print()


def sum_avg_functions():
    print("=" * 60)
    print("2. SUM 求和 与 AVG 平均值")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # SUM: 求和
    cursor.execute("SELECT SUM(amount) AS total_amount FROM sales")
    print(f"总销售额: {cursor.fetchone()['total_amount']:.2f}")

    cursor.execute("SELECT SUM(quantity) AS total_quantity FROM sales")
    print(f"总销售数量: {cursor.fetchone()['total_quantity']}")

    # AVG: 平均值
    cursor.execute("SELECT AVG(amount) AS avg_amount FROM sales")
    print(f"平均单笔金额: {cursor.fetchone()['avg_amount']:.2f}")

    # 结合使用
    cursor.execute("""
        SELECT
            SUM(amount) AS 总销售额,
            SUM(quantity) AS 总销量,
            AVG(amount) AS 平均金额,
            AVG(quantity) AS 平均数量
        FROM sales
    """)
    row = cursor.fetchone()
    print(f"\n综合统计:")
    print(f"  总销售额: {row['总销售额']:.2f}")
    print(f"  总销量: {row['总销量']}")
    print(f"  平均金额: {row['平均金额']:.2f}")
    print(f"  平均数量: {row['平均数量']:.2f}")

    conn.close()
    print()


def max_min_functions():
    print("=" * 60)
    print("3. MAX 最大值 与 MIN 最小值")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            MAX(amount) AS 最高金额,
            MIN(amount) AS 最低金额,
            MAX(quantity) AS 最大数量,
            MIN(quantity) AS 最小数量
        FROM sales
    """)
    row = cursor.fetchone()
    print(f"最高金额: {row['最高金额']}")
    print(f"最低金额: {row['最低金额']}")
    print(f"最大数量: {row['最大数量']}")
    print(f"最小数量: {row['最小数量']}")

    # 配合子查询找出最高金额对应的记录
    cursor.execute("""
        SELECT product, amount, region, sale_date
        FROM sales
        WHERE amount = (SELECT MAX(amount) FROM sales)
    """)
    print("\n最高金额的订单:")
    for row in cursor.fetchall():
        print(f"  {row['product']}: {row['amount']} ({row['region']}, {row['sale_date']})")

    conn.close()
    print()


def group_by():
    print("=" * 60)
    print("4. GROUP BY 分组查询")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 按品类分组
    cursor.execute("""
        SELECT
            category AS 品类,
            COUNT(*) AS 订单数,
            SUM(amount) AS 总销售额,
            AVG(amount) AS 平均金额,
            SUM(quantity) AS 总销量
        FROM sales
        GROUP BY category
    """)
    print("按品类分组统计:")
    print(f"  {'品类':<8} {'订单数':<6} {'总销售额':<12} {'平均金额':<12} {'总销量':<6}")
    for row in cursor.fetchall():
        print(f"  {row['品类']:<8} {row['订单数']:<6} {row['总销售额']:<12.2f} {row['平均金额']:<12.2f} {row['总销量']:<6}")

    # 按地区分组
    cursor.execute("""
        SELECT
            region AS 地区,
            COUNT(*) AS 订单数,
            SUM(amount) AS 总销售额
        FROM sales
        GROUP BY region
        ORDER BY 总销售额 DESC
    """)
    print("\n按地区分组统计 (按销售额降序):")
    for row in cursor.fetchall():
        print(f"  {row['地区']}: {row['订单数']}笔, 总额 {row['总销售额']:.2f}")

    # 多列分组
    cursor.execute("""
        SELECT
            category AS 品类,
            region AS 地区,
            COUNT(*) AS 订单数,
            SUM(amount) AS 总销售额
        FROM sales
        GROUP BY category, region
        ORDER BY category, 总销售额 DESC
    """)
    print("\n按品类+地区分组:")
    print(f"  {'品类':<8} {'地区':<6} {'订单数':<6} {'总销售额':<12}")
    for row in cursor.fetchall():
        print(f"  {row['品类']:<8} {row['地区']:<6} {row['订单数']:<6} {row['总销售额']:<12.2f}")

    conn.close()
    print()


def having_clause():
    print("=" * 60)
    print("5. HAVING 分组后筛选")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # WHERE vs HAVING 的区别:
    # WHERE 在分组前筛选行 (不能使用聚合函数)
    # HAVING 在分组后筛选组 (可以使用聚合函数)

    # 找出总销售额超过5000的品类
    cursor.execute("""
        SELECT
            category AS 品类,
            SUM(amount) AS 总销售额,
            COUNT(*) AS 订单数
        FROM sales
        GROUP BY category
        HAVING SUM(amount) > 5000
    """)
    print("总销售额 > 5000 的品类:")
    for row in cursor.fetchall():
        print(f"  {row['品类']}: 总额 {row['总销售额']:.2f}, {row['订单数']}笔")

    # 组合 WHERE 和 HAVING
    # 先用WHERE筛选华东地区，再用HAVING筛选总销售额>3000的品类
    cursor.execute("""
        SELECT
            category AS 品类,
            SUM(amount) AS 总销售额,
            COUNT(*) AS 订单数
        FROM sales
        WHERE region = '华东'
        GROUP BY category
        HAVING SUM(amount) > 3000
    """)
    print("\n华东地区中总销售额 > 3000 的品类:")
    for row in cursor.fetchall():
        print(f"  {row['品类']}: 总额 {row['总销售额']:.2f}, {row['订单数']}笔")

    # HAVING 多条件
    cursor.execute("""
        SELECT
            product AS 产品,
            SUM(quantity) AS 总销量,
            SUM(amount) AS 总销售额
        FROM sales
        GROUP BY product
        HAVING SUM(quantity) > 10 AND SUM(amount) > 500
    """)
    print("\n总销量>10 且 总销售额>500 的产品:")
    for row in cursor.fetchall():
        print(f"  {row['产品']}: 销量 {row['总销量']}, 总额 {row['总销售额']:.2f}")

    conn.close()
    print()


def aggregate_with_null():
    print("=" * 60)
    print("6. 聚合函数与 NULL 值")
    print("=" * 60)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE test_null (
            id INTEGER PRIMARY KEY,
            name TEXT,
            score REAL
        )
    """)

    cursor.executemany(
        "INSERT INTO test_null (name, score) VALUES (?, ?)",
        [
            ("张三", 85.0),
            ("李四", None),
            ("王五", 92.0),
            ("赵六", None),
            ("孙七", 78.0),
            (None, 88.0),
        ],
    )
    conn.commit()

    # COUNT(*) vs COUNT(column) 对NULL的不同处理
    cursor.execute("""
        SELECT
            COUNT(*) AS 总行数,
            COUNT(name) AS name非空数,
            COUNT(score) AS score非空数
        FROM test_null
    """)
    row = cursor.fetchone()
    print(f"COUNT(*): {row['总行数']}")
    print(f"COUNT(name): {row['name非空数']} (排除NULL)")
    print(f"COUNT(score): {row['score非空数']} (排除NULL)")

    # SUM/AVG 自动忽略NULL值
    cursor.execute("""
        SELECT
            SUM(score) AS 总分,
            AVG(score) AS 平均分,
            MAX(score) AS 最高分,
            MIN(score) AS 最低分
        FROM test_null
    """)
    row = cursor.fetchone()
    print(f"\nSUM(score): {row['总分']} (忽略NULL)")
    print(f"AVG(score): {row['平均分']:.2f} (忽略NULL，只计算非空值)")
    print(f"MAX(score): {row['最高分']}")
    print(f"MIN(score): {row['最低分']}")

    # IFNULL / COALESCE 处理NULL
    cursor.execute("""
        SELECT
            name,
            score,
            IFNULL(score, 0) AS score_替换0,
            COALESCE(score, 60.0) AS score_替换60
        FROM test_null
    """)
    print("\nNULL值替换:")
    print(f"  {'姓名':<6} {'原始分数':<10} {'IFNULL(0)':<10} {'COALESCE(60)':<12}")
    for row in cursor.fetchall():
        name = row['name'] if row['name'] else 'NULL'
        score = row['score'] if row['score'] is not None else 'NULL'
        print(f"  {name:<6} {str(score):<10} {row['score_替换0']:<10} {row['score_替换60']:<12}")

    conn.close()
    print()


if __name__ == "__main__":
    count_function()
    sum_avg_functions()
    max_min_functions()
    group_by()
    having_clause()
    aggregate_with_null()
    print("所有聚合函数与分组查询演示完成！")
