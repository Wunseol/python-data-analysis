# 数据来源: 自建模拟数据与SQLite内存数据库
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


@dataclass
class LoadResult:
    table: str
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_skipped: int = 0
    index_created: list = None

    def __post_init__(self):
        if self.index_created is None:
            self.index_created = []


class WarehouseLoader:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._created_indexes: set[str] = set()

    def _table_exists(self, table: str) -> bool:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None

    def _get_existing_columns(self, table: str) -> list[str]:
        if not self._table_exists(table):
            return []
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]

    def create_table(self, df: pd.DataFrame, table: str) -> None:
        if self._table_exists(table):
            return
        type_map = {
            "int64": "INTEGER", "float64": "REAL", "object": "TEXT",
            "bool": "INTEGER", "datetime64[ns]": "TEXT",
        }
        cols_def = []
        for col in df.columns:
            sql_type = type_map.get(str(df[col].dtype), "TEXT")
            cols_def.append(f'"{col}" {sql_type}')
        sql = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(cols_def)})'
        self.conn.execute(sql)
        self.conn.commit()
        print(f"  [建表] {table}: ({', '.join(df.columns)})")

    def append(self, df: pd.DataFrame, table: str) -> LoadResult:
        self.create_table(df, table)
        rows_before = pd.read_sql(f"SELECT COUNT(*) as c FROM \"{table}\"", self.conn).iloc[0, 0]
        df.to_sql(table, self.conn, if_exists="append", index=False)
        rows_after = pd.read_sql(f"SELECT COUNT(*) as c FROM \"{table}\"", self.conn).iloc[0, 0]
        inserted = rows_after - rows_before
        print(f"  [Append] {table}: 插入 {inserted} 行")
        return LoadResult(table=table, rows_inserted=inserted)

    def upsert(self, df: pd.DataFrame, table: str, key_columns: list[str]) -> LoadResult:
        self.create_table(df, table)

        if not self._table_exists(table):
            df.to_sql(table, self.conn, if_exists="replace", index=False)
            print(f"  [Upsert] {table}: 首次写入 {len(df)} 行")
            return LoadResult(table=table, rows_inserted=len(df))

        existing = pd.read_sql(f'SELECT * FROM "{table}"', self.conn)
        existing_keys = set(existing[key_columns].astype(str).agg("|".join, axis=1))
        new_keys = set(df[key_columns].astype(str).agg("|".join, axis=1))

        to_insert = df[~df[key_columns].astype(str).agg("|".join, axis=1).isin(existing_keys)]
        to_update = df[df[key_columns].astype(str).agg("|".join, axis=1).isin(existing_keys)]

        # 插入新记录
        if len(to_insert) > 0:
            to_insert.to_sql(table, self.conn, if_exists="append", index=False)

        # 更新已有记录
        updated_count = 0
        if len(to_update) > 0:
            for _, row in to_update.iterrows():
                where_clause = " AND ".join([f'"{k}" = ?' for k in key_columns])
                set_clause = ", ".join([f'"{c}" = ?' for c in df.columns if c not in key_columns])
                set_vals = [row[c] for c in df.columns if c not in key_columns]
                key_vals = [row[k] for k in key_columns]
                self.conn.execute(
                    f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}',
                    set_vals + key_vals,
                )
                updated_count += 1
            self.conn.commit()

        print(f"  [Upsert] {table}: 新增 {len(to_insert)}, 更新 {updated_count}, 跳过 {len(df) - len(to_insert) - updated_count}")
        return LoadResult(table=table, rows_inserted=len(to_insert), rows_updated=updated_count)

    def replace(self, df: pd.DataFrame, table: str) -> LoadResult:
        self.create_table(df, table)
        df.to_sql(table, self.conn, if_exists="replace", index=False)
        print(f"  [Replace] {table}: 替换写入 {len(df)} 行")
        return LoadResult(table=table, rows_inserted=len(df))

    def batch_load(self, df: pd.DataFrame, table: str, batch_size: int = 100,
                   mode: Literal["append", "replace"] = "append") -> LoadResult:
        self.create_table(df, table)
        total_inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batch.to_sql(table, self.conn, if_exists=mode, index=False)
            total_inserted += len(batch)
            print(f"  [Batch {i // batch_size + 1}] 写入 {len(batch)} 行")
        print(f"  [Batch完成] {table}: 共写入 {total_inserted} 行")
        return LoadResult(table=table, rows_inserted=total_inserted)

    def create_index(self, table: str, columns: list[str], index_name: str | None = None) -> None:
        if index_name is None:
            index_name = f"idx_{'_'.join(columns)}"
        if index_name in self._created_indexes:
            return
        try:
            cols = ", ".join([f'"{c}"' for c in columns])
            self.conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ({cols})')
            self.conn.commit()
            self._created_indexes.add(index_name)
            print(f"  [Index] {index_name} on {table}({', '.join(columns)})")
        except sqlite3.OperationalError as e:
            print(f"  [Index] 创建失败: {e}")

    def get_table_info(self, table: str) -> None:
        if not self._table_exists(table):
            print(f"  表 '{table}' 不存在")
            return
        count = pd.read_sql(f'SELECT COUNT(*) as c FROM "{table}"', self.conn).iloc[0, 0]
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        columns = [(row[1], row[2]) for row in cur.fetchall()]
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
        )
        indexes = [row[0] for row in cur.fetchall()]
        print(f"  表: {table}")
        print(f"  行数: {count}")
        print(f"  列: {columns}")
        print(f"  索引: {indexes}")


def main():
    conn = sqlite3.connect(":memory:")
    loader = WarehouseLoader(conn)

    # 模拟数据
    np.random.seed(42)
    df_orders = pd.DataFrame({
        "order_id": range(1, 101),
        "customer_id": np.random.randint(1, 30, 100),
        "product": np.random.choice(["手机", "电脑", "耳机", "平板"], 100),
        "amount": np.random.uniform(50, 8000, 100).round(2),
        "order_date": pd.date_range("2025-01-01", periods=100, freq="12h"),
    })

    df_customers = pd.DataFrame({
        "customer_id": range(1, 31),
        "name": [f"客户{i}" for i in range(1, 31)],
        "city": np.random.choice(["北京", "上海", "广州", "深圳"], 30),
        "level": np.random.choice(["普通", "银卡", "金卡"], 30),
    })

    # 1. Append写入
    print("=" * 50)
    print("1. Append 写入")
    print("=" * 50)
    loader.append(df_orders.head(50), "orders")
    loader.append(df_orders.tail(50), "orders")

    # 2. Replace写入
    print("\n" + "=" * 50)
    print("2. Replace 写入")
    print("=" * 50)
    loader.replace(df_customers, "customers")

    # 3. Upsert写入
    print("\n" + "=" * 50)
    print("3. Upsert 写入")
    print("=" * 50)
    df_update = pd.DataFrame({
        "customer_id": [1, 2, 31, 32],
        "name": ["客户1_更新", "客户2_更新", "新客户31", "新客户32"],
        "city": ["杭州", "成都", "武汉", "南京"],
        "level": ["金卡", "金卡", "普通", "银卡"],
    })
    loader.upsert(df_update, "customers", key_columns=["customer_id"])

    # 4. 批量写入
    print("\n" + "=" * 50)
    print("4. 批量写入")
    print("=" * 50)
    df_large = pd.DataFrame({
        "id": range(1, 501),
        "value": np.random.randn(500).round(4),
        "category": np.random.choice(["A", "B", "C"], 500),
    })
    loader.batch_load(df_large, "metrics", batch_size=100, mode="append")

    # 5. 创建索引
    print("\n" + "=" * 50)
    print("5. 创建索引")
    print("=" * 50)
    loader.create_index("orders", ["customer_id"])
    loader.create_index("orders", ["order_date"], "idx_order_date")
    loader.create_index("customers", ["customer_id"], "idx_cust_id")
    loader.create_index("metrics", ["category"], "idx_metrics_cat")

    # 6. 查看表信息
    print("\n" + "=" * 50)
    print("6. 数据仓库概览")
    print("=" * 50)
    for table in ["orders", "customers", "metrics"]:
        loader.get_table_info(table)
        print()

    # 验证数据
    print("=" * 50)
    print("数据验证查询")
    print("=" * 50)
    print("\n订单金额Top5:")
    print(pd.read_sql("SELECT * FROM orders ORDER BY amount DESC LIMIT 5", conn).to_string(index=False))
    print("\n客户等级分布:")
    print(pd.read_sql("SELECT level, COUNT(*) as cnt FROM customers GROUP BY level", conn).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
