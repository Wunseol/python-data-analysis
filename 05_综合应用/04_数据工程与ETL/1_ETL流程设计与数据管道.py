# 数据来源: 自建模拟数据
# 依赖库最低版本要求: pandas>=2.0, sqlalchemy>=2.0, numpy>=1.24
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path
from datetime import datetime


@dataclass
class PipelineContext:
    records_in: int = 0
    records_out: int = 0
    errors: list = field(default_factory=list)
    start_time: datetime = None
    end_time: datetime = None
    metadata: dict = field(default_factory=dict)


class Extractor(ABC):
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        ...


class Transformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class Loader(ABC):
    @abstractmethod
    def load(self, df: pd.DataFrame) -> None:
        ...


class MockExtractor(Extractor):
    def __init__(self, name: str, rows: int = 100):
        self.name = name
        self.rows = rows

    def extract(self) -> pd.DataFrame:
        np.random.seed(42)
        df = pd.DataFrame({
            "order_id": range(1, self.rows + 1),
            "product": np.random.choice(["手机", "电脑", "耳机", "平板"], self.rows),
            "amount": np.random.uniform(50, 8000, self.rows).round(2),
            "quantity": np.random.randint(1, 10, self.rows),
            "order_date": pd.date_range("2025-01-01", periods=self.rows, freq="h"),
        })
        print(f"[Extract] {self.name}: 抽取 {len(df)} 条记录")
        return df


class FilterTransformer(Transformer):
    def __init__(self, column: str, min_value: float):
        self.column = column
        self.min_value = min_value

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[df[self.column] >= self.min_value].copy()
        print(f"[Transform] 过滤 {self.column}>={self.min_value}: {len(df)} -> {len(result)} 条")
        return result


class AddColumnTransformer(Transformer):
    def __init__(self, col_name: str, func: Callable):
        self.col_name = col_name
        self.func = func

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result[self.col_name] = self.func(result)
        print(f"[Transform] 添加列 '{self.col_name}'")
        return result


class ConsoleLoader(Loader):
    def __init__(self, name: str = "console"):
        self.name = name

    def load(self, df: pd.DataFrame) -> None:
        print(f"[Load] {self.name}: 加载 {len(df)} 条记录")
        print(df.head(10).to_string(index=False))


class ETLPipeline:
    def __init__(self, name: str):
        self.name = name
        self.extractors: list[Extractor] = []
        self.transformers: list[Transformer] = []
        self.loaders: list[Loader] = []
        self.context = PipelineContext()

    def add_extractor(self, extractor: Extractor) -> "ETLPipeline":
        self.extractors.append(extractor)
        return self

    def add_transformer(self, transformer: Transformer) -> "ETLPipeline":
        self.transformers.append(transformer)
        return self

    def add_loader(self, loader: Loader) -> "ETLPipeline":
        self.loaders.append(loader)
        return self

    def run(self) -> PipelineContext:
        self.context.start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"ETL管道: {self.name}")
        print(f"{'='*60}")

        # Extract
        print(f"\n--- Extract 阶段 ---")
        frames = []
        for ext in self.extractors:
            try:
                frames.append(ext.extract())
            except Exception as e:
                self.context.errors.append(f"Extract错误: {e}")
        if not frames:
            print("无数据抽取")
            return self.context
        df = pd.concat(frames, ignore_index=True)
        self.context.records_in = len(df)

        # Transform
        print(f"\n--- Transform 阶段 ---")
        for t in self.transformers:
            try:
                df = t.transform(df)
            except Exception as e:
                self.context.errors.append(f"Transform错误: {e}")
        self.context.records_out = len(df)

        # Load
        print(f"\n--- Load 阶段 ---")
        for loader in self.loaders:
            try:
                loader.load(df)
            except Exception as e:
                self.context.errors.append(f"Load错误: {e}")

        self.context.end_time = datetime.now()
        self._print_summary()
        return self.context

    def _print_summary(self):
        duration = (self.context.end_time - self.context.start_time).total_seconds()
        print(f"\n--- 执行摘要 ---")
        print(f"管道名称: {self.name}")
        print(f"输入记录: {self.context.records_in}")
        print(f"输出记录: {self.context.records_out}")
        print(f"过滤比例: {1 - self.context.records_out / max(self.context.records_in, 1):.1%}")
        print(f"执行耗时: {duration:.3f}s")
        print(f"错误数量: {len(self.context.errors)}")
        if self.context.errors:
            for err in self.context.errors:
                print(f"  - {err}")


def demo_dataflow_graph():
    print("\n" + "=" * 60)
    print("数据流图 (DAG)")
    print("=" * 60)

    graph = {
        "csv_orders": ["staging"],
        "api_products": ["staging"],
        "staging": ["clean"],
        "clean": ["enrich"],
        "enrich": ["warehouse"],
        "warehouse": ["report"],
    }

    print("\n节点依赖关系:")
    for src, targets in graph.items():
        for tgt in targets:
            print(f"  {src} ──▶ {tgt}")

    print("\n拓扑执行顺序:")
    visited = set()
    order = []

    def visit(node):
        if node in visited:
            return
        for dep in graph.get(node, []):
            visit(dep)
        visited.add(node)
        order.append(node)

    for node in graph:
        visit(node)

    for i, node in enumerate(order, 1):
        print(f"  {i}. {node}")


def main():
    # 基本ETL管道
    pipeline = ETLPipeline("订单数据处理")
    pipeline.add_extractor(MockExtractor("订单源", 50))
    pipeline.add_transformer(FilterTransformer("amount", 500))
    pipeline.add_transformer(AddColumnTransformer("total", lambda df: df["amount"] * df["quantity"]))
    pipeline.add_loader(ConsoleLoader("控制台输出"))
    pipeline.run()

    # 多抽取器管道
    pipeline2 = ETLPipeline("多源合并")
    pipeline2.add_extractor(MockExtractor("线上订单", 30))
    pipeline2.add_extractor(MockExtractor("线下订单", 20))
    pipeline2.add_transformer(AddColumnTransformer("source", lambda df: "merged"))
    pipeline2.add_loader(ConsoleLoader("合并输出"))
    pipeline2.run()

    # 数据流图
    demo_dataflow_graph()


if __name__ == "__main__":
    main()
