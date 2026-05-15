# 数据源: SQLite内存数据库 (无需外部数据库配置)
# [注意] 本系列使用 SQLAlchemy 2.0 语法风格，与 1.x 有较大差异

"""
综合案例: 电商数据库分析
- 创建完整的电商数据库模型 (users, products, orders, order_items)
- 填充模拟数据
- 执行分析查询 (月度营收、热销商品、用户购买分析)
- 导出结果到 Pandas DataFrame 进行可视化
"""

import sqlite3
import pandas as pd
from sqlalchemy import create_engine, String, Integer, Float, ForeignKey, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship


# ============================================================
# 第一部分: 使用 SQLAlchemy ORM 定义模型并创建数据库
# ============================================================

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(100), nullable=False)
    city: Mapped[str | None] = mapped_column(String(50))
    register_date: Mapped[str] = mapped_column(String(20))

    orders: Mapped[list["Order"]] = relationship(back_populates="user")

    def __repr__(self):
        return f"User(id={self.id}, username='{self.username}', city='{self.city}')"


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)

    order_items: Mapped[list["OrderItem"]] = relationship(back_populates="product")

    def __repr__(self):
        return f"Product(id={self.id}, name='{self.name}', price={self.price})"


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    order_date: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="completed")

    user: Mapped["User"] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order")

    def __repr__(self):
        return f"Order(id={self.id}, user_id={self.user_id}, date='{self.order_date}')"


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)

    order: Mapped["Order"] = relationship(back_populates="items")
    product: Mapped["Product"] = relationship(back_populates="order_items")

    def __repr__(self):
        return f"OrderItem(order_id={self.order_id}, product_id={self.product_id}, qty={self.quantity})"


def create_and_populate_database():
    print("=" * 60)
    print("1. 创建电商数据库并填充模拟数据")
    print("=" * 60)

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # 用户数据
        users = [
            User(username="alice", email="alice@example.com", city="北京", register_date="2023-06-15"),
            User(username="bob", email="bob@example.com", city="上海", register_date="2023-08-20"),
            User(username="charlie", email="charlie@example.com", city="广州", register_date="2023-09-10"),
            User(username="diana", email="diana@example.com", city="深圳", register_date="2023-10-05"),
            User(username="eve", email="eve@example.com", city="杭州", register_date="2023-11-12"),
            User(username="frank", email="frank@example.com", city="北京", register_date="2024-01-08"),
            User(username="grace", email="grace@example.com", city="成都", register_date="2024-02-14"),
            User(username="henry", email="henry@example.com", city="武汉", register_date="2024-03-22"),
        ]
        session.add_all(users)

        # 商品数据
        products = [
            Product(name="笔记本电脑", category="电子产品", price=5999.0, cost=4200.0),
            Product(name="无线鼠标", category="电子产品", price=89.9, cost=35.0),
            Product(name="机械键盘", category="电子产品", price=299.0, cost=120.0),
            Product(name="显示器", category="电子产品", price=1899.0, cost=1100.0),
            Product(name="运动鞋", category="服装", price=399.0, cost=160.0),
            Product(name="羽绒服", category="服装", price=899.0, cost=350.0),
            Product(name="牛仔裤", category="服装", price=259.0, cost=90.0),
            Product(name="咖啡豆", category="食品", price=68.0, cost=25.0),
            Product(name="巧克力", category="食品", price=35.0, cost=12.0),
            Product(name="坚果礼盒", category="食品", price=128.0, cost=55.0),
        ]
        session.add_all(products)

        session.commit()

        # 订单数据
        orders_data = [
            (1, "2024-01-05", "completed"),
            (2, "2024-01-12", "completed"),
            (3, "2024-01-20", "completed"),
            (1, "2024-02-03", "completed"),
            (4, "2024-02-15", "completed"),
            (5, "2024-02-28", "completed"),
            (2, "2024-03-05", "completed"),
            (6, "2024-03-12", "completed"),
            (7, "2024-03-20", "completed"),
            (1, "2024-03-25", "completed"),
            (3, "2024-04-02", "completed"),
            (4, "2024-04-10", "completed"),
            (8, "2024-04-15", "completed"),
            (5, "2024-04-22", "completed"),
            (2, "2024-05-01", "completed"),
            (6, "2024-05-08", "completed"),
            (7, "2024-05-15", "cancelled"),
            (1, "2024-05-20", "completed"),
            (3, "2024-05-28", "completed"),
            (8, "2024-06-05", "completed"),
        ]
        for user_id, order_date, status in orders_data:
            session.add(Order(user_id=user_id, order_date=order_date, status=status))
        session.commit()

        # 订单明细数据
        order_items_data = [
            (1, 1, 1, 5999.0), (1, 2, 2, 89.9),
            (2, 5, 1, 399.0), (2, 6, 1, 899.0),
            (3, 3, 1, 299.0), (3, 8, 3, 68.0),
            (4, 4, 1, 1899.0), (4, 2, 1, 89.9),
            (5, 5, 2, 399.0), (5, 7, 1, 259.0),
            (6, 9, 2, 35.0), (6, 10, 1, 128.0),
            (7, 1, 1, 5999.0), (7, 3, 1, 299.0),
            (8, 6, 1, 899.0), (8, 8, 2, 68.0),
            (9, 2, 3, 89.9), (9, 9, 5, 35.0),
            (10, 4, 1, 1899.0),
            (11, 5, 1, 399.0), (11, 10, 2, 128.0),
            (12, 1, 1, 5999.0), (12, 2, 2, 89.9),
            (13, 3, 1, 299.0), (13, 7, 2, 259.0),
            (14, 6, 1, 899.0), (14, 8, 3, 68.0),
            (15, 4, 1, 1899.0), (15, 5, 1, 399.0),
            (16, 9, 3, 35.0), (16, 10, 2, 128.0),
            (17, 1, 1, 5999.0),
            (18, 3, 2, 299.0), (18, 8, 2, 68.0),
            (19, 5, 1, 399.0), (19, 6, 1, 899.0), (19, 7, 1, 259.0),
            (20, 2, 2, 89.9), (20, 9, 5, 35.0), (20, 10, 1, 128.0),
        ]
        for order_id, product_id, quantity, unit_price in order_items_data:
            session.add(OrderItem(order_id=order_id, product_id=product_id, quantity=quantity, unit_price=unit_price))
        session.commit()

        print(f"已创建数据库:")
        print(f"  用户: {session.query(User).count()} 条")
        print(f"  商品: {session.query(Product).count()} 条")
        print(f"  订单: {session.query(Order).count()} 条")
        print(f"  订单明细: {session.query(OrderItem).count()} 条")

    return engine


def monthly_revenue_analysis(engine):
    print("=" * 60)
    print("2. 月度营收分析")
    print("=" * 60)

    query = text("""
        SELECT
            strftime('%Y-%m', o.order_date) AS month,
            COUNT(DISTINCT o.id) AS order_count,
            SUM(oi.quantity) AS total_quantity,
            SUM(oi.quantity * oi.unit_price) AS revenue,
            SUM(oi.quantity * (oi.unit_price - p.cost)) AS profit
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        WHERE o.status = 'completed'
        GROUP BY strftime('%Y-%m', o.order_date)
        ORDER BY month
    """)

    with engine.connect() as conn:
        df_monthly = pd.read_sql_query(query, conn)

    print("月度营收统计:")
    print(df_monthly.to_string(index=False))

    print(f"\n总营收: ¥{df_monthly['revenue'].sum():,.2f}")
    print(f"总利润: ¥{df_monthly['profit'].sum():,.2f}")
    print(f"利润率: {df_monthly['profit'].sum() / df_monthly['revenue'].sum() * 100:.1f}%")

    return df_monthly


def top_products_analysis(engine):
    print("=" * 60)
    print("3. 热销商品分析")
    print("=" * 60)

    query = text("""
        SELECT
            p.name AS product_name,
            p.category,
            SUM(oi.quantity) AS total_sold,
            SUM(oi.quantity * oi.unit_price) AS total_revenue,
            SUM(oi.quantity * (oi.unit_price - p.cost)) AS total_profit,
            ROUND(SUM(oi.quantity * (oi.unit_price - p.cost)) / SUM(oi.quantity * oi.unit_price) * 100, 1) AS profit_rate
        FROM order_items oi
        JOIN products p ON oi.product_id = p.id
        JOIN orders o ON oi.order_id = o.id
        WHERE o.status = 'completed'
        GROUP BY p.id, p.name, p.category
        ORDER BY total_revenue DESC
    """)

    with engine.connect() as conn:
        df_products = pd.read_sql_query(query, conn)

    print("商品销售排行:")
    print(df_products.to_string(index=False))

    print("\n品类销售汇总:")
    category_summary = df_products.groupby("category").agg(
        total_revenue=("total_revenue", "sum"),
        total_profit=("total_profit", "sum"),
        total_sold=("total_sold", "sum"),
    ).sort_values("total_revenue", ascending=False)
    print(category_summary.to_string())

    return df_products


def user_purchase_analysis(engine):
    print("=" * 60)
    print("4. 用户购买分析")
    print("=" * 60)

    query = text("""
        SELECT
            u.username,
            u.city,
            COUNT(DISTINCT o.id) AS order_count,
            SUM(oi.quantity) AS total_items,
            SUM(oi.quantity * oi.unit_price) AS total_spent,
            AVG(oi.quantity * oi.unit_price) AS avg_item_value,
            MAX(o.order_date) AS last_order_date
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id AND o.status = 'completed'
        LEFT JOIN order_items oi ON o.id = oi.order_id
        GROUP BY u.id, u.username, u.city
        ORDER BY total_spent DESC
    """)

    with engine.connect() as conn:
        df_users = pd.read_sql_query(query, conn)

    print("用户消费排行:")
    print(df_users.to_string(index=False))

    print("\n城市消费汇总:")
    city_summary = df_users.groupby("city").agg(
        user_count=("username", "count"),
        total_spent=("total_spent", "sum"),
        avg_spent=("total_spent", "mean"),
    ).sort_values("total_spent", ascending=False)
    print(city_summary.to_string())

    return df_users


def category_monthly_analysis(engine):
    print("=" * 60)
    print("5. 品类月度趋势分析")
    print("=" * 60)

    query = text("""
        SELECT
            strftime('%Y-%m', o.order_date) AS month,
            p.category,
            SUM(oi.quantity) AS total_sold,
            SUM(oi.quantity * oi.unit_price) AS revenue
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        WHERE o.status = 'completed'
        GROUP BY strftime('%Y-%m', o.order_date), p.category
        ORDER BY month, p.category
    """)

    with engine.connect() as conn:
        df_cat_monthly = pd.read_sql_query(query, conn)

    # 透视表: 行=月份, 列=品类, 值=营收
    pivot = df_cat_monthly.pivot_table(
        index="month",
        columns="category",
        values="revenue",
        fill_value=0,
    )
    print("品类月度营收透视表:")
    print(pivot.to_string())

    # 透视表: 行=月份, 列=品类, 值=销量
    pivot_qty = df_cat_monthly.pivot_table(
        index="month",
        columns="category",
        values="total_sold",
        fill_value=0,
    )
    print("\n品类月度销量透视表:")
    print(pivot_qty.to_string())

    return df_cat_monthly


def order_status_analysis(engine):
    print("=" * 60)
    print("6. 订单状态分析")
    print("=" * 60)

    query = text("""
        SELECT
            o.status,
            COUNT(*) AS order_count,
            COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_value
        FROM orders o
        LEFT JOIN order_items oi ON o.id = oi.order_id
        GROUP BY o.status
    """)

    with engine.connect() as conn:
        df_status = pd.read_sql_query(query, conn)

    print("订单状态统计:")
    print(df_status.to_string(index=False))

    total_orders = df_status["order_count"].sum()
    completed_rate = df_status[df_status["status"] == "completed"]["order_count"].values[0] / total_orders * 100
    print(f"\n订单完成率: {completed_rate:.1f}%")

    return df_status


def comprehensive_report(engine):
    print("=" * 60)
    print("7. 综合分析报告")
    print("=" * 60)

    with engine.connect() as conn:
        # 关键指标
        kpi_query = text("""
            SELECT
                (SELECT COUNT(*) FROM users) AS total_users,
                (SELECT COUNT(*) FROM orders WHERE status = 'completed') AS completed_orders,
                (SELECT COUNT(*) FROM products) AS total_products,
                COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue,
                COALESCE(SUM(oi.quantity * (oi.unit_price - p.cost)), 0) AS total_profit
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.status = 'completed'
        """)
        df_kpi = pd.read_sql_query(kpi_query, conn)

    print("核心业务指标:")
    kpi = df_kpi.iloc[0]
    print(f"  总用户数: {int(kpi['total_users'])}")
    print(f"  完成订单数: {int(kpi['completed_orders'])}")
    print(f"  商品种类: {int(kpi['total_products'])}")
    print(f"  总营收: ¥{kpi['total_revenue']:,.2f}")
    print(f"  总利润: ¥{kpi['total_profit']:,.2f}")
    if kpi['total_revenue'] > 0:
        print(f"  综合利润率: {kpi['total_profit'] / kpi['total_revenue'] * 100:.1f}%")
    print(f"  客单价: ¥{kpi['total_revenue'] / kpi['completed_orders']:,.2f}")
    print(f"  人均消费: ¥{kpi['total_revenue'] / kpi['total_users']:,.2f}")

    # RFM分析概念 (最近购买时间、购买频率、消费金额)
    print("\nRFM分析 (用户价值分层):")
    rfm_query = text("""
        SELECT
            u.username,
            MAX(o.order_date) AS recency,
            COUNT(DISTINCT o.id) AS frequency,
            SUM(oi.quantity * oi.unit_price) AS monetary
        FROM users u
        JOIN orders o ON u.id = o.user_id AND o.status = 'completed'
        JOIN order_items oi ON o.id = oi.order_id
        GROUP BY u.id, u.username
        ORDER BY monetary DESC
    """)
    with engine.connect() as conn:
        df_rfm = pd.read_sql_query(rfm_query, conn)

    print(df_rfm.to_string(index=False))

    return df_kpi


def export_to_dataframe_demo(engine):
    print("=" * 60)
    print("8. 导出分析结果到 DataFrame 进行进一步处理")
    print("=" * 60)

    with engine.connect() as conn:
        # 将所有表导出为DataFrame
        df_users = pd.read_sql_table("users", conn)
        df_products = pd.read_sql_table("products", conn)
        df_orders = pd.read_sql_table("orders", conn)
        df_order_items = pd.read_sql_table("order_items", conn)

    print("各表导出为DataFrame:")
    for name, df in [("users", df_users), ("products", df_products), ("orders", df_orders), ("order_items", df_order_items)]:
        print(f"  {name}: {df.shape[0]}行 x {df.shape[1]}列, 列名={list(df.columns)}")

    # 在DataFrame中进行合并分析
    df_merged = df_order_items.merge(df_orders, left_on="order_id", right_on="id", suffixes=("_item", "_order"))
    df_merged = df_merged.merge(df_products, left_on="product_id", right_on="id")
    df_merged = df_merged.merge(df_users, left_on="user_id", right_on="id")

    print(f"\n合并后大表: {df_merged.shape[0]}行 x {df_merged.shape[1]}列")
    print(f"列名: {list(df_merged.columns)}")

    # 使用Pandas进行分析 (与SQL分析对比)
    print("\nPandas方式 - 各品类营收:")
    category_rev = df_merged[df_merged["status"] == "completed"].groupby("category").agg(
        revenue=("quantity", lambda x: (x * df_merged.loc[x.index, "unit_price"]).sum()),
        quantity=("quantity", "sum"),
    ).sort_values("revenue", ascending=False)
    print(category_rev)

    print("\nPandas方式 - 各城市消费:")
    city_spend = df_merged[df_merged["status"] == "completed"].groupby("city").agg(
        total_spent=("quantity", lambda x: (x * df_merged.loc[x.index, "unit_price"]).sum()),
        order_count=("order_id", "nunique"),
    ).sort_values("total_spent", ascending=False)
    print(city_spend)

    return df_merged


if __name__ == "__main__":
    engine = create_and_populate_database()
    df_monthly = monthly_revenue_analysis(engine)
    df_products = top_products_analysis(engine)
    df_users = user_purchase_analysis(engine)
    df_cat_monthly = category_monthly_analysis(engine)
    df_status = order_status_analysis(engine)
    df_kpi = comprehensive_report(engine)
    df_merged = export_to_dataframe_demo(engine)
    print("\n综合案例: 电商数据库分析 完成！")
