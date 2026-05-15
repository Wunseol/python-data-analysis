# 数据来源: 自建模拟数据(CSV/Excel/JSON/SQLite)
import pandas as pd
import numpy as np
import sqlite3
import json
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class DataSourceMeta:
    name: str
    source_type: str
    schema_info: dict = field(default_factory=dict)
    row_count: int = 0
    columns: list = field(default_factory=list)


class DataSourceRegistry:
    def __init__(self):
        self._sources: dict[str, DataSourceMeta] = {}

    def register(self, name: str, source_type: str, **kwargs) -> None:
        meta = DataSourceMeta(name=name, source_type=source_type, **kwargs)
        self._sources[name] = meta
        print(f"[注册] 数据源 '{name}' (类型: {source_type})")

    def get(self, name: str) -> DataSourceMeta | None:
        return self._sources.get(name)

    def list_sources(self) -> list[DataSourceMeta]:
        return list(self._sources.values())

    def remove(self, name: str) -> None:
        if name in self._sources:
            del self._sources[name]
            print(f"[注销] 数据源 '{name}'")


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def metadata(self) -> DataSourceMeta:
        ...


class CSVExtractor(BaseExtractor):
    def __init__(self, name: str, file_path: Path):
        self.name = name
        self.file_path = file_path

    def extract(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        print(f"[CSV抽取] {self.name}: {len(df)} 行, {len(df.columns)} 列")
        return df

    def metadata(self) -> DataSourceMeta:
        return DataSourceMeta(
            name=self.name, source_type="csv",
            columns=list(pd.read_csv(self.file_path, nrows=0).columns),
        )


class ExcelExtractor(BaseExtractor):
    def __init__(self, name: str, file_path: Path, sheet_name: str | int = 0):
        self.name = name
        self.file_path = file_path
        self.sheet_name = sheet_name

    def extract(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, engine="openpyxl")
        print(f"[Excel抽取] {self.name}: {len(df)} 行, {len(df.columns)} 列")
        return df

    def metadata(self) -> DataSourceMeta:
        return DataSourceMeta(
            name=self.name, source_type="excel",
            columns=list(pd.read_excel(self.file_path, nrows=0, engine="openpyxl").columns),
        )


class JSONExtractor(BaseExtractor):
    def __init__(self, name: str, file_path: Path, orient: str = "records"):
        self.name = name
        self.file_path = file_path
        self.orient = orient

    def extract(self) -> pd.DataFrame:
        df = pd.read_json(self.file_path, orient=self.orient, lines=True)
        print(f"[JSON抽取] {self.name}: {len(df)} 行, {len(df.columns)} 列")
        return df

    def metadata(self) -> DataSourceMeta:
        df_sample = pd.read_json(self.file_path, orient=self.orient, lines=True, nrows=1)
        return DataSourceMeta(
            name=self.name, source_type="json",
            columns=list(df_sample.columns),
        )


class SQLiteExtractor(BaseExtractor):
    def __init__(self, name: str, conn: sqlite3.Connection, table: str):
        self.name = name
        self.conn = conn
        self.table = table

    def extract(self) -> pd.DataFrame:
        df = pd.read_sql(f"SELECT * FROM {self.table}", self.conn)
        print(f"[SQLite抽取] {self.name}: {len(df)} 行, {len(df.columns)} 列")
        return df

    def metadata(self) -> DataSourceMeta:
        cols_df = pd.read_sql(f"SELECT * FROM {self.table} LIMIT 0", self.conn)
        count_df = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {self.table}", self.conn)
        return DataSourceMeta(
            name=self.name, source_type="sqlite",
            columns=list(cols_df.columns),
            row_count=int(count_df.iloc[0, 0]),
        )


class UnifiedExtractor:
    def __init__(self, registry: DataSourceRegistry):
        self.registry = registry
        self._extractors: dict[str, BaseExtractor] = {}

    def register_extractor(self, name: str, extractor: BaseExtractor) -> None:
        self._extractors[name] = extractor
        meta = extractor.metadata()
        meta.row_count = len(extractor.extract())
        self.registry.register(name, meta.source_type, columns=meta.columns, row_count=meta.row_count)

    def extract(self, name: str) -> pd.DataFrame:
        if name not in self._extractors:
            raise ValueError(f"未注册的数据源: {name}")
        return self._extractors[name].extract()

    def extract_all(self) -> dict[str, pd.DataFrame]:
        results = {}
        for name, ext in self._extractors.items():
            results[name] = ext.extract()
        return results


def prepare_sample_data(base_dir: Path) -> dict[str, Path]:
    np.random.seed(42)
    paths = {}

    # CSV
    csv_path = base_dir / "sample_orders.csv"
    df_csv = pd.DataFrame({
        "order_id": range(1, 51),
        "customer": [f"客户{i}" for i in range(1, 51)],
        "product": np.random.choice(["手机", "电脑", "耳机"], 50),
        "amount": np.random.uniform(100, 5000, 50).round(2),
    })
    df_csv.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    # Excel
    excel_path = base_dir / "sample_products.xlsx"
    df_excel = pd.DataFrame({
        "product_id": range(1, 21),
        "product_name": [f"商品{i}" for i in range(1, 21)],
        "category": np.random.choice(["电子", "家居", "食品"], 20),
        "price": np.random.uniform(10, 1000, 20).round(2),
    })
    df_excel.to_excel(excel_path, index=False, engine="openpyxl")
    paths["excel"] = excel_path

    # JSON (lines格式)
    json_path = base_dir / "sample_customers.json"
    records = [
        {"id": i, "name": f"用户{i}", "city": np.random.choice(["北京", "上海", "广州", "深圳"])}
        for i in range(1, 31)
    ]
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    json_path.write_text(lines, encoding="utf-8")
    paths["json"] = json_path

    return paths


def prepare_sqlite_data() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    np.random.seed(42)
    df = pd.DataFrame({
        "log_id": range(1, 41),
        "action": np.random.choice(["登录", "浏览", "购买", "退出"], 40),
        "user_id": np.random.randint(1, 20, 40),
        "timestamp": pd.date_range("2025-01-01", periods=40, freq="6h"),
    })
    df.to_sql("user_logs", conn, index=False, if_exists="replace")
    return conn


def main():
    base_dir = Path(__file__).parent / "_temp_data"
    base_dir.mkdir(exist_ok=True)

    # 准备模拟数据
    paths = prepare_sample_data(base_dir)
    sqlite_conn = prepare_sqlite_data()

    # 创建注册中心和统一抽取器
    registry = DataSourceRegistry()
    unified = UnifiedExtractor(registry)

    # 注册各数据源
    unified.register_extractor("orders_csv", CSVExtractor("orders_csv", paths["csv"]))
    unified.register_extractor("products_excel", ExcelExtractor("products_excel", paths["excel"]))
    unified.register_extractor("customers_json", JSONExtractor("customers_json", paths["json"]))
    unified.register_extractor("logs_sqlite", SQLiteExtractor("logs_sqlite", sqlite_conn, "user_logs"))

    # 查看注册信息
    print("\n" + "=" * 50)
    print("已注册数据源")
    print("=" * 50)
    for meta in registry.list_sources():
        print(f"  {meta.name} | 类型: {meta.source_type} | 行数: {meta.row_count} | 列: {meta.columns}")

    # 统一抽取
    print("\n" + "=" * 50)
    print("统一抽取所有数据源")
    print("=" * 50)
    all_data = unified.extract_all()
    for name, df in all_data.items():
        print(f"\n[{name}]")
        print(df.head(5).to_string(index=False))

    # 单源抽取
    print("\n" + "=" * 50)
    print("单源抽取示例")
    print("=" * 50)
    df_orders = unified.extract("orders_csv")
    print(f"订单数据: {df_orders.shape}")
    print(df_orders.dtypes)

    # 清理临时文件
    for p in paths.values():
        p.unlink(missing_ok=True)
    (base_dir / "sample_products.xlsx").unlink(missing_ok=True)
    base_dir.rmdir()
    sqlite_conn.close()


if __name__ == "__main__":
    main()
