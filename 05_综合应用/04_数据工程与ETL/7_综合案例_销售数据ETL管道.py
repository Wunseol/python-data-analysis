# 数据来源: 自建模拟销售数据与SQLite内存数据库
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from typing import Callable
from abc import ABC, abstractmethod


# ============================================================
# 基础框架
# ============================================================

class TaskStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    name: str
    status: TaskStatus
    duration: float = 0.0
    records_in: int = 0
    records_out: int = 0
    error: str | None = None


class TransformStep(ABC):
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass
class ValidationRule:
    name: str
    func: Callable[[pd.DataFrame], tuple[bool, str]]


# ============================================================
# 数据源模拟
# ============================================================

def create_online_orders(n: int = 200) -> pd.DataFrame:
    np.random.seed(100)
    return pd.DataFrame({
        "order_id": [f"ON{1000+i}" for i in range(n)],
        "customer_id": np.random.randint(1, 50, n),
        "product_id": np.random.randint(1, 20, n),
        "quantity": np.random.randint(1, 5, n),
        "unit_price": np.random.uniform(50, 3000, n).round(2),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="8h"),
        "channel": "线上",
        "status": np.random.choice(["已完成", "已取消", "待发货"], n, p=[0.8, 0.1, 0.1]),
        "updated_at": pd.date_range("2025-01-01", periods=n, freq="8h"),
    })


def create_offline_orders(n: int = 100) -> pd.DataFrame:
    np.random.seed(200)
    return pd.DataFrame({
        "order_id": [f"OF{2000+i}" for i in range(n)],
        "customer_id": np.random.randint(1, 50, n),
        "product_id": np.random.randint(1, 20, n),
        "quantity": np.random.randint(1, 10, n),
        "unit_price": np.random.uniform(30, 2000, n).round(2),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="1D"),
        "channel": "线下",
        "status": np.random.choice(["已完成", "已退货"], n, p=[0.9, 0.1]),
        "updated_at": pd.date_range("2025-01-01", periods=n, freq="1D"),
    })


def create_product_master() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": range(1, 21),
        "product_name": [
            "智能手机A", "笔记本电脑B", "无线耳机C", "平板电脑D", "智能手表E",
            "蓝牙音箱F", "键盘G", "鼠标H", "显示器I", "摄像头J",
            "路由器K", "充电宝L", "数据线M", "手机壳N", "贴膜O",
            "散热器P", "硬盘Q", "内存条R", "显卡S", "主板T",
        ],
        "category": ["电子"] * 8 + ["配件"] * 7 + ["硬件"] * 5,
        "cost_ratio": np.random.uniform(0.3, 0.7, 20).round(2),
    })


def create_customer_master() -> pd.DataFrame:
    np.random.seed(300)
    return pd.DataFrame({
        "customer_id": range(1, 51),
        "customer_name": [f"客户{i:03d}" for i in range(1, 51)],
        "city": np.random.choice(["北京", "上海", "广州", "深圳", "杭州", "成都"], 50),
        "level": np.random.choice(["普通", "银卡", "金卡", "钻石"], 50, p=[0.4, 0.3, 0.2, 0.1]),
        "register_date": pd.date_range("2024-01-01", periods=50, freq="3D"),
    })


# ============================================================
# ETL步骤实现
# ============================================================

class SalesETL:
    def __init__(self):
        self.warehouse_conn = sqlite3.connect(":memory:")
        self.source_conn = sqlite3.connect(":memory:")
        self.results: list[TaskResult] = []
        self.last_etl_time: datetime | None = None
        self._setup_logging()
        self._init_source()

    def _setup_logging(self):
        self.logger = logging.getLogger("SalesETL")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
            self.logger.addHandler(handler)

    def _init_source(self):
        create_online_orders().to_sql("online_orders", self.source_conn, index=False, if_exists="replace")
        create_offline_orders().to_sql("offline_orders", self.source_conn, index=False, if_exists="replace")
        create_product_master().to_sql("products", self.source_conn, index=False, if_exists="replace")
        create_customer_master().to_sql("customers", self.source_conn, index=False, if_exists="replace")
        self.source_conn.commit()

    def _record_result(self, name: str, status: TaskStatus, start: datetime, end: datetime,
                       records_in: int = 0, records_out: int = 0, error: str | None = None):
        result = TaskResult(
            name=name, status=status,
            duration=(end - start).total_seconds(),
            records_in=records_in, records_out=records_out, error=error,
        )
        self.results.append(result)
        icon = "✓" if status == TaskStatus.SUCCESS else "✗"
        self.logger.info(f"{icon} {name}: {status.value} ({result.duration:.2f}s) [{records_in}→{records_out}]")
        return result

    # --- Extract ---
    def extract(self) -> dict[str, pd.DataFrame]:
        start = datetime.now()
        try:
            online = pd.read_sql("SELECT * FROM online_orders", self.source_conn)
            offline = pd.read_sql("SELECT * FROM offline_orders", self.source_conn)
            products = pd.read_sql("SELECT * FROM products", self.source_conn)
            customers = pd.read_sql("SELECT * FROM customers", self.source_conn)

            total = len(online) + len(offline) + len(products) + len(customers)
            self._record_result("Extract", TaskStatus.SUCCESS, start, datetime.now(),
                                records_in=total, records_out=total)
            return {"online": online, "offline": offline, "products": products, "customers": customers}
        except Exception as e:
            self._record_result("Extract", TaskStatus.FAILED, start, datetime.now(), error=str(e))
            raise

    # --- Transform ---
    def transform(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        start = datetime.now()
        try:
            total_in = sum(len(df) for df in data.values())

            # 合并线上线下订单
            orders = pd.concat([data["online"], data["offline"]], ignore_index=True)

            # 关联商品和客户
            orders = orders.merge(data["products"], on="product_id", how="left")
            orders = orders.merge(data["customers"], on="customer_id", how="left")

            # 计算金额
            orders["total_amount"] = orders["quantity"] * orders["unit_price"]
            orders["cost_amount"] = (orders["total_amount"] * orders["cost_ratio"]).round(2)
            orders["profit"] = (orders["total_amount"] - orders["cost_amount"]).round(2)

            # 金额分级
            orders["amount_level"] = pd.cut(
                orders["total_amount"], bins=[0, 500, 2000, 5000, 100000],
                labels=["低", "中", "高", "极高"],
            )

            # 清洗: 去除取消和退货订单(仅保留已完成和待发货)
            valid = orders[orders["status"].isin(["已完成", "待发货"])].copy()

            # 去重
            valid = valid.drop_duplicates(subset=["order_id"]).reset_index(drop=True)

            # 选择最终列
            final_cols = [
                "order_id", "customer_id", "customer_name", "city", "level",
                "product_id", "product_name", "category",
                "quantity", "unit_price", "total_amount", "cost_amount", "profit",
                "amount_level", "channel", "status", "order_date", "updated_at",
            ]
            valid = valid[[c for c in final_cols if c in valid.columns]]

            self._record_result("Transform", TaskStatus.SUCCESS, start, datetime.now(),
                                records_in=total_in, records_out=len(valid))
            return valid
        except Exception as e:
            self._record_result("Transform", TaskStatus.FAILED, start, datetime.now(), error=str(e))
            raise

    # --- Load ---
    def load(self, df: pd.DataFrame) -> None:
        start = datetime.now()
        try:
            df.to_sql("fact_orders", self.warehouse_conn, if_exists="replace", index=False)

            # 创建维度表
            dim_product = df[["product_id", "product_name", "category"]].drop_duplicates()
            dim_product.to_sql("dim_product", self.warehouse_conn, if_exists="replace", index=False)

            dim_customer = df[["customer_id", "customer_name", "city", "level"]].drop_duplicates()
            dim_customer.to_sql("dim_customer", self.warehouse_conn, if_exists="replace", index=False)

            # 创建索引
            self.warehouse_conn.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON fact_orders(order_date)")
            self.warehouse_conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON fact_orders(channel)")
            self.warehouse_conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON fact_orders(category)")
            self.warehouse_conn.commit()

            self._record_result("Load", TaskStatus.SUCCESS, start, datetime.now(),
                                records_in=len(df), records_out=len(df))
        except Exception as e:
            self._record_result("Load", TaskStatus.FAILED, start, datetime.now(), error=str(e))
            raise

    # --- 增量更新 ---
    def incremental_update(self) -> None:
        start = datetime.now()
        try:
            if self.last_etl_time is None:
                print("\n  (首次运行, 无增量更新)")
                self._record_result("Incremental", TaskStatus.SKIPPED, start, datetime.now())
                return

            ts = self.last_etl_time.strftime("%Y-%m-%d %H:%M:%S")
            new_online = pd.read_sql(
                f"SELECT * FROM online_orders WHERE updated_at > '{ts}'", self.source_conn
            )
            new_offline = pd.read_sql(
                f"SELECT * FROM offline_orders WHERE updated_at > '{ts}'", self.source_conn
            )

            if len(new_online) == 0 and len(new_offline) == 0:
                print(f"  无增量数据 (>{ts})")
                self._record_result("Incremental", TaskStatus.SUCCESS, start, datetime.now())
                return

            # 模拟增量数据
            np.random.seed(999)
            new_records = pd.DataFrame({
                "order_id": [f"INC{3000+i}" for i in range(5)],
                "customer_id": np.random.randint(1, 50, 5),
                "product_id": np.random.randint(1, 20, 5),
                "quantity": np.random.randint(1, 5, 5),
                "unit_price": np.random.uniform(100, 2000, 5).round(2),
                "order_date": pd.Timestamp.now() + pd.to_timedelta(range(5), unit="h"),
                "channel": "线上",
                "status": "已完成",
                "updated_at": pd.Timestamp.now() + pd.to_timedelta(range(5), unit="h"),
            })
            new_records.to_sql("online_orders", self.source_conn, if_exists="append", index=False)

            # 追加到仓库
            products = pd.read_sql("SELECT * FROM products", self.source_conn)
            customers = pd.read_sql("SELECT * FROM customers", self.source_conn)
            merged = new_records.merge(products, on="product_id", how="left")
            merged = merged.merge(customers, on="customer_id", how="left")
            merged["total_amount"] = merged["quantity"] * merged["unit_price"]
            merged["cost_amount"] = (merged["total_amount"] * merged["cost_ratio"]).round(2)
            merged["profit"] = (merged["total_amount"] - merged["cost_amount"]).round(2)
            merged["amount_level"] = pd.cut(
                merged["total_amount"], bins=[0, 500, 2000, 5000, 100000],
                labels=["低", "中", "高", "极高"],
            )

            final_cols = [
                "order_id", "customer_id", "customer_name", "city", "level",
                "product_id", "product_name", "category",
                "quantity", "unit_price", "total_amount", "cost_amount", "profit",
                "amount_level", "channel", "status", "order_date", "updated_at",
            ]
            merged = merged[[c for c in final_cols if c in merged.columns]]
            merged.to_sql("fact_orders", self.warehouse_conn, if_exists="append", index=False)
            self.warehouse_conn.commit()

            self._record_result("Incremental", TaskStatus.SUCCESS, start, datetime.now(),
                                records_in=len(new_records), records_out=len(merged))
        except Exception as e:
            self._record_result("Incremental", TaskStatus.FAILED, start, datetime.now(), error=str(e))

    # --- 校验 ---
    def validate(self) -> None:
        start = datetime.now()
        try:
            df = pd.read_sql("SELECT * FROM fact_orders", self.warehouse_conn)
            rules = [
                ValidationRule("非空检查-order_id", lambda d: (d["order_id"].notna().all(), f"空值: {d['order_id'].isna().sum()}")),
                ValidationRule("唯一性-order_id", lambda d: (d["order_id"].duplicated().sum() == 0, f"重复: {d['order_id'].duplicated().sum()}")),
                ValidationRule("金额非负", lambda d: ((d["total_amount"] >= 0).all(), f"负值: {(d['total_amount'] < 0).sum()}")),
                ValidationRule("利润合理性", lambda d: ((d["profit"] >= -d["total_amount"] * 0.5).all(), "利润异常")),
            ]

            all_passed = True
            for rule in rules:
                passed, detail = rule.func(df)
                icon = "✓" if passed else "✗"
                print(f"  {icon} {rule.name}: {detail}")
                if not passed:
                    all_passed = False

            status = TaskStatus.SUCCESS if all_passed else TaskStatus.FAILED
            self._record_result("Validate", status, start, datetime.now(),
                                records_in=len(df), records_out=len(df))
        except Exception as e:
            self._record_result("Validate", TaskStatus.FAILED, start, datetime.now(), error=str(e))

    # --- 报告 ---
    def generate_report(self) -> None:
        start = datetime.now()
        try:
            df = pd.read_sql("SELECT * FROM fact_orders", self.warehouse_conn)

            print(f"\n{'='*60}")
            print(f"销售数据仓库报告")
            print(f"{'='*60}")
            print(f"总订单数: {len(df)}")
            print(f"总销售额: ¥{df['total_amount'].sum():,.2f}")
            print(f"总利润: ¥{df['profit'].sum():,.2f}")
            print(f"利润率: {df['profit'].sum() / df['total_amount'].sum():.1%}")

            print(f"\n--- 按渠道 ---")
            ch = df.groupby("channel").agg(
                订单数=("order_id", "count"),
                销售额=("total_amount", "sum"),
                利润=("profit", "sum"),
            ).round(2)
            print(ch.to_string())

            print(f"\n--- 按品类 ---")
            cat = df.groupby("category").agg(
                订单数=("order_id", "count"),
                销售额=("total_amount", "sum"),
                利润=("profit", "sum"),
            ).round(2)
            print(cat.to_string())

            print(f"\n--- 按客户等级 ---")
            lvl = df.groupby("level").agg(
                订单数=("order_id", "count"),
                销售额=("total_amount", "sum"),
            ).round(2)
            print(lvl.to_string())

            print(f"\n--- 按城市Top5 ---")
            city = df.groupby("city")["total_amount"].sum().nlargest(5).round(2)
            print(city.to_string())

            self._record_result("Report", TaskStatus.SUCCESS, start, datetime.now(),
                                records_in=len(df), records_out=len(df))
        except Exception as e:
            self._record_result("Report", TaskStatus.FAILED, start, datetime.now(), error=str(e))

    # --- 完整运行 ---
    def run_full(self):
        print(f"\n{'#'*60}")
        print(f"# 销售数据ETL管道 - 全量执行")
        print(f"{'#'*60}")
        self.results.clear()
        self.last_etl_time = None

        data = self.extract()
        transformed = self.transform(data)
        self.load(transformed)
        self.last_etl_time = datetime.now()
        self.incremental_update()
        self.validate()
        self.generate_report()
        self._print_summary()

    def run_incremental(self):
        print(f"\n{'#'*60}")
        print(f"# 销售数据ETL管道 - 增量执行")
        print(f"{'#'*60}")
        self.results.clear()

        self.incremental_update()
        self.validate()
        self.generate_report()
        self._print_summary()

    def _print_summary(self):
        print(f"\n{'='*60}")
        print(f"ETL执行摘要")
        print(f"{'='*60}")
        total_duration = 0.0
        for r in self.results:
            icon = "✓" if r.status == TaskStatus.SUCCESS else ("⊘" if r.status == TaskStatus.SKIPPED else "✗")
            line = f"  {icon} {r.name}: {r.status.value}"
            if r.duration > 0:
                line += f" ({r.duration:.2f}s)"
            if r.records_in > 0:
                line += f" [{r.records_in}→{r.records_out}]"
            if r.error:
                line += f" - {r.error}"
            print(line)
            total_duration += r.duration

        success = sum(1 for r in self.results if r.status == TaskStatus.SUCCESS)
        print(f"\n  总耗时: {total_duration:.2f}s | 成功: {success}/{len(self.results)}")

    def close(self):
        self.warehouse_conn.close()
        self.source_conn.close()


# ============================================================
# DAG工作流编排
# ============================================================

class WorkflowDAG:
    def __init__(self, name: str):
        self.name = name
        self.tasks: dict[str, Callable] = {}
        self.dependencies: dict[str, list[str]] = {}

    def add_task(self, name: str, func: Callable, depends_on: list[str] | None = None):
        self.tasks[name] = func
        self.dependencies[name] = depends_on or []

    def run(self):
        order = self._topological_sort()
        print(f"\n工作流: {self.name}")
        print(f"执行顺序: {' → '.join(order)}")

        completed = set()
        for name in order:
            for dep in self.dependencies[name]:
                if dep not in completed:
                    print(f"  跳过 {name} (依赖 {dep} 未完成)")
                    break
            else:
                print(f"\n▶ 执行: {name}")
                try:
                    self.tasks[name]()
                    completed.add(name)
                except Exception as e:
                    print(f"  失败: {e}")
                    break

    def _topological_sort(self) -> list[str]:
        in_degree = {n: 0 for n in self.tasks}
        graph = {n: [] for n in self.tasks}
        for name, deps in self.dependencies.items():
            for dep in deps:
                if dep in self.tasks:
                    graph[dep].append(name)
                    in_degree[name] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order


def main():
    etl = SalesETL()

    # 全量ETL
    etl.run_full()

    # 增量ETL
    etl.run_incremental()

    # DAG工作流编排
    print(f"\n{'#'*60}")
    print(f"# DAG工作流编排执行")
    print(f"{'#'*60}")

    etl2 = SalesETL()
    dag = WorkflowDAG("销售ETL自动化")

    data_holder = {}

    def step_extract():
        data_holder["data"] = etl2.extract()

    def step_transform():
        data_holder["transformed"] = etl2.transform(data_holder["data"])

    def step_load():
        etl2.load(data_holder["transformed"])
        etl2.last_etl_time = datetime.now()

    def step_incremental():
        etl2.incremental_update()

    def step_validate():
        etl2.validate()

    def step_report():
        etl2.generate_report()

    dag.add_task("extract", step_extract)
    dag.add_task("transform", step_transform, depends_on=["extract"])
    dag.add_task("load", step_load, depends_on=["transform"])
    dag.add_task("incremental", step_incremental, depends_on=["load"])
    dag.add_task("validate", step_validate, depends_on=["incremental"])
    dag.add_task("report", step_report, depends_on=["validate"])
    dag.run()

    etl2._print_summary()

    etl.close()
    etl2.close()


if __name__ == "__main__":
    main()
