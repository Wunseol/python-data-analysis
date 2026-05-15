# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
多表连接 JOIN
- INNER JOIN 内连接
- LEFT JOIN 左连接
- 自连接 (Self-Join) 概念
- 创建多个关联表
- 连接条件与复合连接
"""

import sqlite3


def setup_database():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 部门表
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT
        )
    """)

    # 员工表 (外键关联部门)
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department_id INTEGER,
            salary REAL NOT NULL,
            manager_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES departments(id),
            FOREIGN KEY (manager_id) REFERENCES employees(id)
        )
    """)

    # 项目表
    cursor.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department_id INTEGER,
            budget REAL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)

    # 插入部门数据
    departments = [
        ("技术部", "北京"),
        ("市场部", "上海"),
        ("财务部", "广州"),
        ("人事部", "深圳"),
    ]
    cursor.executemany("INSERT INTO departments (name, location) VALUES (?, ?)", departments)

    # 插入员工数据 (含 manager_id 用于自连接演示)
    employees = [
        ("张三", 1, 25000.0, None),
        ("李四", 1, 20000.0, 1),
        ("王五", 2, 18000.0, None),
        ("赵六", 2, 15000.0, 3),
        ("孙七", 3, 16000.0, None),
        ("周八", 3, 14000.0, 5),
        ("吴九", 1, 22000.0, 1),
        ("郑十", None, 12000.0, None),
    ]
    cursor.executemany(
        "INSERT INTO employees (name, department_id, salary, manager_id) VALUES (?, ?, ?, ?)",
        employees,
    )

    # 插入项目数据
    projects = [
        ("电商平台重构", 1, 500000.0),
        ("品牌推广计划", 2, 200000.0),
        ("财务系统升级", 3, 300000.0),
        ("AI助手开发", 1, 800000.0),
    ]
    cursor.executemany(
        "INSERT INTO projects (name, department_id, budget) VALUES (?, ?, ?)",
        projects,
    )

    conn.commit()
    return conn


def inner_join():
    print("=" * 60)
    print("1. INNER JOIN 内连接")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # INNER JOIN: 只返回两表中匹配的行
    cursor.execute("""
        SELECT
            e.name AS 员工姓名,
            d.name AS 部门名称,
            d.location AS 部门地点,
            e.salary AS 薪资
        FROM employees e
        INNER JOIN departments d ON e.department_id = d.id
    """)
    print("员工-部门内连接:")
    print(f"  {'员工姓名':<8} {'部门名称':<8} {'部门地点':<8} {'薪资':<10}")
    for row in cursor.fetchall():
        print(f"  {row['员工姓名']:<8} {row['部门名称']:<8} {row['部门地点']:<8} {row['薪资']:<10}")

    # 注意: 郑十没有部门(department_id=NULL)，不会出现在结果中
    print("\n注意: 郑十(department_id=NULL)未出现在结果中")

    conn.close()
    print()


def left_join():
    print("=" * 60)
    print("2. LEFT JOIN 左连接")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # LEFT JOIN: 返回左表所有行，右表无匹配则为NULL
    cursor.execute("""
        SELECT
            e.name AS 员工姓名,
            d.name AS 部门名称,
            d.location AS 部门地点,
            e.salary AS 薪资
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
    """)
    print("员工-部门左连接:")
    print(f"  {'员工姓名':<8} {'部门名称':<10} {'部门地点':<10} {'薪资':<10}")
    for row in cursor.fetchall():
        dept = row['部门名称'] if row['部门名称'] else '(无部门)'
        loc = row['部门地点'] if row['部门地点'] else '-'
        print(f"  {row['员工姓名']:<8} {dept:<10} {loc:<10} {row['薪资']:<10}")

    print("\n注意: 郑十出现在结果中，部门信息为NULL")

    # 左连接找出没有员工的部门
    cursor.execute("""
        SELECT
            d.name AS 部门名称,
            COUNT(e.id) AS 员工数
        FROM departments d
        LEFT JOIN employees e ON d.id = e.department_id
        GROUP BY d.id, d.name
    """)
    print("\n各部门员工数 (包括0人的部门):")
    for row in cursor.fetchall():
        print(f"  {row['部门名称']}: {row['员工数']}人")

    conn.close()
    print()


def multi_table_join():
    print("=" * 60)
    print("3. 多表连接")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 连接员工、部门、项目三张表
    cursor.execute("""
        SELECT
            e.name AS 员工姓名,
            d.name AS 部门名称,
            p.name AS 项目名称,
            p.budget AS 项目预算
        FROM employees e
        INNER JOIN departments d ON e.department_id = d.id
        INNER JOIN projects p ON d.id = p.department_id
        ORDER BY d.name, e.name
    """)
    print("员工-部门-项目 三表连接:")
    print(f"  {'员工姓名':<8} {'部门名称':<8} {'项目名称':<14} {'项目预算':<10}")
    for row in cursor.fetchall():
        print(f"  {row['员工姓名']:<8} {row['部门名称']:<8} {row['项目名称']:<14} {row['项目预算']:<10}")

    conn.close()
    print()


def self_join():
    print("=" * 60)
    print("4. 自连接 (Self-Join)")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 自连接: 员工表自己连接自己，查找员工与其经理的关系
    # manager_id 指向同一张表的 id
    cursor.execute("""
        SELECT
            e.name AS 员工姓名,
            e.salary AS 员工薪资,
            m.name AS 经理姓名,
            m.salary AS 经理薪资
        FROM employees e
        LEFT JOIN employees m ON e.manager_id = m.id
    """)
    print("员工-经理 自连接:")
    print(f"  {'员工姓名':<8} {'员工薪资':<10} {'经理姓名':<10} {'经理薪资':<10}")
    for row in cursor.fetchall():
        mgr_name = row['经理姓名'] if row['经理姓名'] else '(无经理)'
        mgr_salary = row['经理薪资'] if row['经理薪资'] else '-'
        print(f"  {row['员工姓名']:<8} {row['员工薪资']:<10} {mgr_name:<10} {mgr_salary:<10}")

    conn.close()
    print()


def join_with_aggregation():
    print("=" * 60)
    print("5. 连接 + 聚合函数")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # 各部门薪资统计
    cursor.execute("""
        SELECT
            d.name AS 部门名称,
            COUNT(e.id) AS 员工数,
            SUM(e.salary) AS 薪资总额,
            AVG(e.salary) AS 平均薪资,
            MAX(e.salary) AS 最高薪资,
            MIN(e.salary) AS 最低薪资
        FROM departments d
        LEFT JOIN employees e ON d.id = e.department_id
        GROUP BY d.id, d.name
        ORDER BY 薪资总额 DESC
    """)
    print("各部门薪资统计:")
    print(f"  {'部门':<8} {'人数':<4} {'总额':<12} {'平均':<12} {'最高':<10} {'最低':<10}")
    for row in cursor.fetchall():
        count = row['员工数']
        total = row['薪资总额'] if row['薪资总额'] else 0
        avg = row['平均薪资'] if row['平均薪资'] else 0
        max_s = row['最高薪资'] if row['最高薪资'] else 0
        min_s = row['最低薪资'] if row['最低薪资'] else 0
        print(f"  {row['部门名称']:<8} {count:<4} {total:<12.0f} {avg:<12.0f} {max_s:<10.0f} {min_s:<10.0f}")

    # 各部门项目预算统计
    cursor.execute("""
        SELECT
            d.name AS 部门名称,
            COUNT(p.id) AS 项目数,
            SUM(p.budget) AS 总预算,
            AVG(p.budget) AS 平均预算
        FROM departments d
        LEFT JOIN projects p ON d.id = p.department_id
        GROUP BY d.id, d.name
    """)
    print("\n各部门项目预算:")
    for row in cursor.fetchall():
        proj_count = row['项目数']
        total_budget = row['总预算'] if row['总预算'] else 0
        avg_budget = row['平均预算'] if row['平均预算'] else 0
        print(f"  {row['部门名称']}: {proj_count}个项目, 总预算 {total_budget:.0f}, 平均 {avg_budget:.0f}")

    conn.close()
    print()


def join_comparison():
    print("=" * 60)
    print("6. INNER JOIN vs LEFT JOIN 对比")
    print("=" * 60)

    conn = setup_database()
    cursor = conn.cursor()

    # INNER JOIN 结果数
    cursor.execute("""
        SELECT COUNT(*) AS cnt FROM employees e
        INNER JOIN departments d ON e.department_id = d.id
    """)
    inner_count = cursor.fetchone()['cnt']

    # LEFT JOIN 结果数
    cursor.execute("""
        SELECT COUNT(*) AS cnt FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
    """)
    left_count = cursor.fetchone()['cnt']

    print(f"INNER JOIN 结果行数: {inner_count}")
    print(f"LEFT JOIN  结果行数: {left_count}")
    print(f"差异: {left_count - inner_count} 行 (LEFT JOIN多出的NULL匹配行)")

    # 使用 LEFT JOIN 找出没有匹配的行
    cursor.execute("""
        SELECT e.name, e.department_id
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE d.id IS NULL
    """)
    print("\nLEFT JOIN 中右表为NULL的行 (没有部门的员工):")
    for row in cursor.fetchall():
        print(f"  {row['name']} (department_id={row['department_id']})")

    conn.close()
    print()


if __name__ == "__main__":
    inner_join()
    left_join()
    multi_table_join()
    self_join()
    join_with_aggregation()
    join_comparison()
    print("所有多表连接JOIN演示完成！")
