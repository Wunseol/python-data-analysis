# 数据来源: 自建模拟数据与SQLite内存数据库
import pandas as pd
import numpy as np
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class IncrementalMode(Enum):
    TIMESTAMP = "timestamp"
    VERSION = "version"


@dataclass
class IncrementalState:
    last_timestamp: datetime | None = None
    last_version: int = 0


@dataclass
class ValidationResult:
    rule_name: str
    passed: bool
    detail: str = ""
    affected_rows: int = 0


@dataclass
class ValidationReport:
    table_name: str
    total_rows: int
    results: list[ValidationResult] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / max(len(self.results), 1)

    def print_report(self):
        print(f"\n{'='*55}")
        print(f"数据校验报告: {self.table_name}")
        print(f"{'='*55}")
        print(f"校验时间: {self.checked_at:%Y-%m-%d %H:%M:%S}")
        print(f"总行数: {self.total_rows}")
        print(f"通过: {self.passed_count} | 失败: {self.failed_count} | 通过率: {self.pass_rate:.0%}")
        print(f"{'-'*55}")
        for r in self.results:
            status = "✓ 通过" if r.passed else "✗ 失败"
            print(f"  {status} | {r.rule_name}: {r.detail}")
        print(f"{'='*55}")


class IncrementalExtractor:
    def __init__(self, conn: sqlite3.Connection, table: str,
                 mode: IncrementalMode = IncrementalMode.TIMESTAMP,
                 timestamp_col: str = "updated_at",
                 version_col: str = "version"):
        self.conn = conn
        self.table = table
        self.mode = mode
        self.timestamp_col = timestamp_col
        self.version_col = version_col
        self.state = IncrementalState()

    def extract_full(self) -> pd.DataFrame:
        df = pd.read_sql(f'SELECT * FROM "{self.table}"', self.conn)
        print(f"[全量抽取] {self.table}: {len(df)} 行")
        self._update_state(df)
        return df

    def extract_incremental(self) -> pd.DataFrame:
        if self.state.last_timestamp is None and self.state.last_version == 0:
            print(f"[增量抽取] 无历史状态, 执行全量抽取")
            return self.extract_full()

        if self.mode == IncrementalMode.TIMESTAMP:
            ts = self.state.last_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_sql(
                f'SELECT * FROM "{self.table}" WHERE "{self.timestamp_col}" > ?',
                self.conn, params=[ts]
            )
            print(f"[增量抽取-时间戳] {self.table}: 新增/变更 {len(df)} 行 (>{ts})")
        else:
            df = pd.read_sql(
                f'SELECT * FROM "{self.table}" WHERE "{self.version_col}" > ?',
                self.conn, params=[self.state.last_version]
            )
            print(f"[增量抽取-版本号] {self.table}: 新增/变更 {len(df)} 行 (>v{self.state.last_version})")

        self._update_state(df)
        return df

    def _update_state(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        if self.mode == IncrementalMode.TIMESTAMP and self.timestamp_col in df.columns:
            max_ts = pd.to_datetime(df[self.timestamp_col]).max()
            self.state.last_timestamp = max_ts.to_pydatetime()
        elif self.mode == IncrementalMode.VERSION and self.version_col in df.columns:
            self.state.last_version = int(df[self.version_col].max())


class DataValidator:
    def __init__(self):
        self.rules: list = []

    def add_completeness_rule(self, column: str, threshold: float = 1.0):
        self.rules.append(("completeness", column, threshold))

    def add_uniqueness_rule(self, column: str):
        self.rules.append(("uniqueness", column, None))

    def add_range_rule(self, column: str, min_val: float, max_val: float):
        self.rules.append(("range", column, (min_val, max_val)))

    def add_consistency_rule(self, column_a: str, column_b: str, condition: str):
        self.rules.append(("consistency", f"{column_a},{column_b}", condition))

    def add_custom_rule(self, name: str, func):
        self.rules.append(("custom", name, func))

    def validate(self, df: pd.DataFrame, table_name: str = "data") -> ValidationReport:
        report = ValidationReport(table_name=table_name, total_rows=len(df))

        for rule_type, column, param in self.rules:
            if rule_type == "completeness":
                result = self._check_completeness(df, column, param)
            elif rule_type == "uniqueness":
                result = self._check_uniqueness(df, column)
            elif rule_type == "range":
                result = self._check_range(df, column, param)
            elif rule_type == "consistency":
                result = self._check_consistency(df, column, param)
            elif rule_type == "custom":
                result = self._check_custom(df, column, param)
            else:
                result = ValidationResult(rule_name=column, passed=False, detail="未知规则类型")
            report.results.append(result)

        return report

    def _check_completeness(self, df: pd.DataFrame, column: str, threshold: float) -> ValidationResult:
        if column not in df.columns:
            return ValidationResult("完整性", False, f"列 '{column}' 不存在")
        if len(df) == 0:
            return ValidationResult(f"完整性({column})", True, "空数据集, 跳过检查")
        non_null = df[column].notna().sum()
        rate = non_null / len(df)
        passed = rate >= threshold
        return ValidationResult(
            f"完整性({column})", passed,
            f"非空率 {rate:.1%} (阈值 {threshold:.0%})",
            affected_rows=len(df) - non_null,
        )

    def _check_uniqueness(self, df: pd.DataFrame, column: str) -> ValidationResult:
        if column not in df.columns:
            return ValidationResult(f"唯一性({column})", False, f"列 '{column}' 不存在")
        dup_count = df[column].duplicated().sum()
        passed = dup_count == 0
        return ValidationResult(
            f"唯一性({column})", passed,
            f"重复 {dup_count} 行",
            affected_rows=dup_count,
        )

    def _check_range(self, df: pd.DataFrame, column: str, range_val: tuple) -> ValidationResult:
        if column not in df.columns:
            return ValidationResult(f"范围({column})", False, f"列 '{column}' 不存在")
        min_val, max_val = range_val
        numeric = pd.to_numeric(df[column], errors="coerce")
        out_of_range = ((numeric < min_val) | (numeric > max_val)).sum()
        passed = out_of_range == 0
        return ValidationResult(
            f"范围({column})", passed,
            f"超出[{min_val},{max_val}]共 {out_of_range} 行",
            affected_rows=out_of_range,
        )

    def _check_consistency(self, df: pd.DataFrame, columns: str, condition: str) -> ValidationResult:
        col_a, col_b = columns.split(",")
        if col_a not in df.columns or col_b not in df.columns:
            return ValidationResult(f"一致性({col_a},{col_b})", False, "列不存在")
        if condition == "a_gte_b":
            violations = (df[col_a] < df[col_b]).sum()
        elif condition == "a_eq_b":
            violations = (df[col_a] != df[col_b]).sum()
        else:
            violations = 0
        passed = violations == 0
        return ValidationResult(
            f"一致性({col_a},{col_b})", passed,
            f"不一致 {violations} 行",
            affected_rows=violations,
        )

    def _check_custom(self, df: pd.DataFrame, name: str, func) -> ValidationResult:
        try:
            passed, detail = func(df)
            return ValidationResult(f"自定义({name})", passed, detail)
        except Exception as e:
            return ValidationResult(f"自定义({name})", False, f"执行错误: {e}")


def prepare_source_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    np.random.seed(42)
    base_time = datetime(2025, 1, 1)
    records = []
    for i in range(1, 101):
        ts = base_time + timedelta(hours=i * 6)
        records.append({
            "id": i,
            "name": f"商品{i}",
            "price": round(np.random.uniform(10, 5000), 2),
            "stock": np.random.randint(0, 200),
            "version": 1,
            "updated_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
        })
    pd.DataFrame(records).to_sql("products", conn, index=False, if_exists="replace")
    return conn


def simulate_updates(conn: sqlite3.Connection):
    # 模拟新增和更新
    updates = [
        (101, "商品101", 999.99, 50, 1, "2025-03-20 12:00:00"),
        (102, "商品102", 1299.00, 30, 1, "2025-03-20 18:00:00"),
        (1, "商品1_更新", 199.99, 80, 2, "2025-03-21 10:00:00"),
        (5, "商品5_更新", 899.00, 0, 2, "2025-03-21 14:00:00"),
    ]
    for rec in updates:
        conn.execute(
            "INSERT OR REPLACE INTO products (id, name, price, stock, version, updated_at) VALUES (?,?,?,?,?,?)",
            rec,
        )
    conn.commit()
    print(f"[模拟更新] 写入 {len(updates)} 条变更")


def main():
    source_conn = prepare_source_db()

    # === 增量抽取: 时间戳模式 ===
    print("=" * 55)
    print("1. 增量抽取 - 时间戳模式")
    print("=" * 55)
    ext_ts = IncrementalExtractor(source_conn, "products", mode=IncrementalMode.TIMESTAMP)
    df_full = ext_ts.extract_full()
    print(f"  状态: last_timestamp = {ext_ts.state.last_timestamp}")

    simulate_updates(source_conn)
    df_inc = ext_ts.extract_incremental()
    print(f"  增量数据:")
    print(df_inc.to_string(index=False))
    print(f"  状态: last_timestamp = {ext_ts.state.last_timestamp}")

    # === 增量抽取: 版本号模式 ===
    print("\n" + "=" * 55)
    print("2. 增量抽取 - 版本号模式")
    print("=" * 55)
    ext_ver = IncrementalExtractor(source_conn, "products", mode=IncrementalMode.VERSION)
    df_full2 = ext_ver.extract_full()
    print(f"  状态: last_version = {ext_ver.state.last_version}")

    # 模拟更多更新
    conn2 = source_conn
    conn2.execute(
        "INSERT OR REPLACE INTO products VALUES (103, '商品103', 599.0, 100, 2, '2025-04-01 08:00:00')"
    )
    conn2.execute(
        "INSERT OR REPLACE INTO products VALUES (2, '商品2_更新', 450.0, 60, 2, '2025-04-01 10:00:00')"
    )
    conn2.commit()
    df_inc2 = ext_ver.extract_incremental()
    print(f"  增量数据:")
    print(df_inc2.to_string(index=False))
    print(f"  状态: last_version = {ext_ver.state.last_version}")

    # === 全量 vs 增量对比 ===
    print("\n" + "=" * 55)
    print("3. 全量 vs 增量对比")
    print("=" * 55)
    df_all = pd.read_sql("SELECT * FROM products", source_conn)
    print(f"  全量数据: {len(df_all)} 行")
    print(f"  增量数据: {len(df_inc2)} 行")
    print(f"  增量占比: {len(df_inc2)/len(df_all):.1%}")

    # === 数据质量校验 ===
    print("\n" + "=" * 55)
    print("4. 数据质量校验")
    print("=" * 55)
    validator = DataValidator()
    validator.add_completeness_rule("id", 1.0)
    validator.add_completeness_rule("name", 1.0)
    validator.add_completeness_rule("price", 0.95)
    validator.add_uniqueness_rule("id")
    validator.add_range_rule("price", 0, 100000)
    validator.add_range_rule("stock", 0, 10000)
    validator.add_consistency_rule("price", "stock", "a_gte_b")
    validator.add_custom_rule("价格非零", lambda df: (
        (df["price"] > 0).all(),
        f"零价格记录: {(df['price'] <= 0).sum()} 条"
    ))

    report = validator.validate(df_all, "products")
    report.print_report()

    # 校验增量数据
    print("\n校验增量数据:")
    inc_report = validator.validate(df_inc2, "products_incremental")
    inc_report.print_report()

    source_conn.close()


if __name__ == "__main__":
    main()
