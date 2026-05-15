# 数据来源: 自建模拟数据
import pandas as pd
import numpy as np
import sqlite3
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from datetime import datetime
from collections import deque


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    task_name: str
    status: TaskStatus
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    retry_count: int = 0

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class Task:
    name: str
    func: Callable[[], any]
    depends_on: list[str] = field(default_factory=list)
    max_retries: int = 3
    retry_delay: float = 0.1
    result: TaskResult | None = None


class DAG:
    def __init__(self, name: str):
        self.name = name
        self.tasks: dict[str, Task] = {}

    def add_task(self, name: str, func: Callable, depends_on: list[str] | None = None,
                 max_retries: int = 3) -> None:
        task = Task(name=name, func=func, depends_on=depends_on or [], max_retries=max_retries)
        self.tasks[name] = task

    def topological_sort(self) -> list[str]:
        in_degree = {name: 0 for name in self.tasks}
        graph = {name: [] for name in self.tasks}
        for name, task in self.tasks.items():
            for dep in task.depends_on:
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

        if len(order) != len(self.tasks):
            raise ValueError("DAG中存在循环依赖")
        return order

    def print_dag(self):
        print(f"\nDAG: {self.name}")
        print(f"{'─'*40}")
        for name, task in self.tasks.items():
            deps = f" ← {task.depends_on}" if task.depends_on else " (起始)"
            print(f"  {name}{deps}")
        order = self.topological_sort()
        print(f"执行顺序: {' → '.join(order)}")


class WorkflowEngine:
    def __init__(self, dag: DAG, max_retries: int = 3):
        self.dag = dag
        self.max_retries = max_retries
        self.results: list[TaskResult] = []
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(f"workflow_{self.dag.name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(handler)

    def run(self) -> list[TaskResult]:
        order = self.dag.topological_sort()
        self.dag.print_dag()
        print(f"\n开始执行工作流: {self.dag.name}")
        print(f"{'='*50}")

        for task_name in order:
            task = self.dag.tasks[task_name]

            # 检查依赖是否成功
            skip = False
            for dep in task.depends_on:
                dep_result = self.dag.tasks[dep].result
                if dep_result and dep_result.status == TaskStatus.FAILED:
                    skip = True
                    break

            if skip:
                result = TaskResult(
                    task_name=task_name, status=TaskStatus.SKIPPED,
                    start_time=datetime.now(), end_time=datetime.now(),
                    error="依赖任务失败",
                )
                task.result = result
                self.results.append(result)
                self.logger.warning(f"跳过任务: {task_name} (依赖失败)")
                continue

            result = self._execute_task(task)
            task.result = result
            self.results.append(result)

        self._print_summary()
        return self.results

    def _execute_task(self, task: Task) -> TaskResult:
        result = TaskResult(task_name=task.name, status=TaskStatus.RUNNING)
        result.start_time = datetime.now()
        self.logger.info(f"开始任务: {task.name}")

        retries = 0
        last_error = None
        while retries <= task.max_retries:
            try:
                task.func()
                result.status = TaskStatus.SUCCESS
                result.end_time = datetime.now()
                result.retry_count = retries
                self.logger.info(f"任务完成: {task.name} ({result.duration:.2f}s)")
                return result
            except Exception as e:
                retries += 1
                last_error = str(e)
                if retries <= task.max_retries:
                    result.status = TaskStatus.RETRYING
                    self.logger.warning(f"任务失败, 重试 {retries}/{task.max_retries}: {task.name} - {e}")
                    import time
                    time.sleep(task.retry_delay)
                else:
                    result.status = TaskStatus.FAILED
                    result.end_time = datetime.now()
                    result.error = last_error
                    result.retry_count = retries
                    self.logger.error(f"任务最终失败: {task.name} - {e}")
                    return result

    def _print_summary(self):
        print(f"\n{'='*50}")
        print(f"工作流执行摘要: {self.dag.name}")
        print(f"{'='*50}")
        for r in self.results:
            status_icon = {
                TaskStatus.SUCCESS: "✓",
                TaskStatus.FAILED: "✗",
                TaskStatus.SKIPPED: "⊘",
            }.get(r.status, "?")
            line = f"  {status_icon} {r.task_name}: {r.status.value}"
            if r.duration > 0:
                line += f" ({r.duration:.2f}s)"
            if r.retry_count > 0:
                line += f" [重试{r.retry_count}次]"
            if r.error:
                line += f" - {r.error}"
            print(line)

        success = sum(1 for r in self.results if r.status == TaskStatus.SUCCESS)
        total = len(self.results)
        print(f"\n结果: {success}/{total} 成功")


def main():
    conn = sqlite3.connect(":memory:")

    # 定义ETL任务函数
    def task_extract_orders():
        np.random.seed(42)
        df = pd.DataFrame({
            "order_id": range(1, 51),
            "product": np.random.choice(["手机", "电脑", "耳机"], 50),
            "amount": np.random.uniform(100, 5000, 50).round(2),
        })
        df.to_sql("raw_orders", conn, if_exists="replace", index=False)
        print(f"    抽取订单: {len(df)} 条")

    def task_extract_products():
        df = pd.DataFrame({
            "product_id": range(1, 11),
            "product_name": [f"商品{i}" for i in range(1, 11)],
            "category": np.random.choice(["电子", "家居"], 10),
        })
        df.to_sql("raw_products", conn, if_exists="replace", index=False)
        print(f"    抽取商品: {len(df)} 条")

    def task_clean_orders():
        df = pd.read_sql("SELECT * FROM raw_orders", conn)
        df = df.drop_duplicates()
        df["amount"] = df["amount"].fillna(0)
        df.to_sql("clean_orders", conn, if_exists="replace", index=False)
        print(f"    清洗订单: {len(df)} 条")

    def task_clean_products():
        df = pd.read_sql("SELECT * FROM raw_products", conn)
        df = df.drop_duplicates()
        df.to_sql("clean_products", conn, if_exists="replace", index=False)
        print(f"    清洗商品: {len(df)} 条")

    def task_transform():
        orders = pd.read_sql("SELECT * FROM clean_orders", conn)
        products = pd.read_sql("SELECT * FROM clean_products", conn)
        merged = orders.merge(products, left_on="product", right_on="product_name", how="left")
        merged["amount_level"] = pd.cut(merged["amount"], bins=[0, 500, 2000, 10000],
                                         labels=["低", "中", "高"])
        merged.to_sql("transformed", conn, if_exists="replace", index=False)
        print(f"    转换合并: {len(merged)} 条")

    def task_load():
        df = pd.read_sql("SELECT * FROM transformed", conn)
        df.to_sql("warehouse_fact", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON warehouse_fact(category)")
        conn.commit()
        print(f"    加载到仓库: {len(df)} 条")

    def task_validate():
        df = pd.read_sql("SELECT * FROM warehouse_fact", conn)
        assert len(df) > 0, "仓库数据为空"
        assert df["amount"].notna().all(), "存在空金额"
        print(f"    校验通过: {len(df)} 条")

    # 可能失败的任务(模拟)
    fail_counter = {"count": 0}

    def task_report():
        df = pd.read_sql("SELECT * FROM warehouse_fact", conn)
        summary = df.groupby("category")["amount"].agg(["count", "mean", "sum"]).round(2)
        print(f"    报告生成:")
        print(summary.to_string())

    def task_notify():
        print(f"    通知: ETL流程完成")

    # 构建DAG
    dag = DAG("销售数据ETL")
    dag.add_task("extract_orders", task_extract_orders)
    dag.add_task("extract_products", task_extract_products)
    dag.add_task("clean_orders", task_clean_orders, depends_on=["extract_orders"])
    dag.add_task("clean_products", task_clean_products, depends_on=["extract_products"])
    dag.add_task("transform", task_transform, depends_on=["clean_orders", "clean_products"])
    dag.add_task("load", task_load, depends_on=["transform"])
    dag.add_task("validate", task_validate, depends_on=["load"])
    dag.add_task("report", task_report, depends_on=["validate"])
    dag.add_task("notify", task_notify, depends_on=["report"])

    # 执行工作流
    engine = WorkflowEngine(dag)
    results = engine.run()

    # 演示: 带失败重试的DAG
    print("\n\n" + "=" * 50)
    print("演示: 任务失败与重试")
    print("=" * 50)

    attempt = {"n": 0}

    def flaky_task():
        attempt["n"] += 1
        if attempt["n"] < 3:
            raise RuntimeError(f"模拟失败 (第{attempt['n']}次)")
        print(f"    第{attempt['n']}次尝试成功!")

    dag2 = DAG("重试演示")
    dag2.add_task("flaky", flaky_task, max_retries=3)
    engine2 = WorkflowEngine(dag2)
    engine2.run()

    # 演示: 依赖失败导致跳过
    print("\n\n" + "=" * 50)
    print("演示: 依赖失败 → 后续跳过")
    print("=" * 50)

    def always_fail():
        raise RuntimeError("故意失败")

    def downstream_task():
        print("    这不会执行")

    dag3 = DAG("失败传播演示")
    dag3.add_task("fail_task", always_fail, max_retries=1)
    dag3.add_task("skip_task", downstream_task, depends_on=["fail_task"])
    engine3 = WorkflowEngine(dag3)
    engine3.run()

    conn.close()


if __name__ == "__main__":
    main()
