# 数据来源: 自建模拟数据
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable


class TransformStep(ABC):
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class RenameStep(TransformStep):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.rename(columns=self.mapping)
        renamed = [f"{k}→{v}" for k, v in self.mapping.items() if k in df.columns]
        print(f"  [Rename] {', '.join(renamed)}")
        return result


class CastStep(TransformStep):
    def __init__(self, column: str, dtype: str):
        self.column = column
        self.dtype = dtype

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        old_dtype = str(result[self.column].dtype)
        result[self.column] = result[self.column].astype(self.dtype)
        print(f"  [Cast] {self.column}: {old_dtype} → {self.dtype}")
        return result


class FilterStep(TransformStep):
    def __init__(self, condition: Callable[[pd.DataFrame], pd.Series], desc: str = ""):
        self.condition = condition
        self.desc = desc

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = self.condition(df)
        result = df[mask].copy()
        print(f"  [Filter] {self.desc or '自定义条件'}: {len(df)} → {len(result)} 行")
        return result


class DeduplicateStep(TransformStep):
    def __init__(self, subset: list[str] | None = None, keep: str = "first"):
        self.subset = subset
        self.keep = keep

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        result = df.drop_duplicates(subset=self.subset, keep=self.keep).reset_index(drop=True)
        dup_count = before - len(result)
        print(f"  [Deduplicate] 去除 {dup_count} 条重复, 保留 {len(result)} 行")
        return result


class FillNAStep(TransformStep):
    def __init__(self, strategy: dict[str, Any]):
        self.strategy = strategy

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, val in self.strategy.items():
            if col in result.columns:
                na_count = result[col].isna().sum()
                if val == "mean":
                    result[col] = result[col].fillna(result[col].mean())
                elif val == "median":
                    result[col] = result[col].fillna(result[col].median())
                else:
                    result[col] = result[col].fillna(val)
                print(f"  [FillNA] {col}: 填充 {na_count} 个空值 (策略: {val})")
        return result


class CustomStep(TransformStep):
    def __init__(self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.name = name
        self.func = func

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        result = self.func(df)
        print(f"  [Custom:{self.name}] {before} → {len(result)} 行")
        return result


class TransformPipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps: list[TransformStep] = []

    def add(self, step: TransformStep) -> "TransformPipeline":
        self.steps.append(step)
        return self

    def chain(self, *steps: TransformStep) -> "TransformPipeline":
        self.steps.extend(steps)
        return self

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n转换管道 [{self.name}] 开始, 输入: {len(df)} 行")
        result = df
        for i, step in enumerate(self.steps, 1):
            print(f"  步骤 {i}/{len(self.steps)}:")
            result = step.execute(result)
        print(f"转换管道 [{self.name}] 完成, 输出: {len(result)} 行")
        return result


def create_dirty_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "user_id": list(range(1, n // 2 + 1)) * 2,
        "UserName": [f"user_{i}" for i in range(n)],
        "AGE": np.random.choice([np.nan, 18, 25, 30, 35, 40, 45, 50], n),
        "Email": [f"user{i}@test.com" if i % 7 != 0 else None for i in range(n)],
        "order_amt": np.random.choice([np.nan, 0, 50, 100, 200, 500, 1000], n).astype(float),
        "city": np.random.choice(["北京", "上海", "广州", None, "深圳", ""], n),
    })
    return df


def main():
    df = create_dirty_data()
    print("原始脏数据:")
    print(df.head(10).to_string(index=False))
    print(f"\n空值统计:\n{df.isna().sum()}")

    # 管道1: 基础清洗
    pipeline1 = TransformPipeline("基础清洗")
    pipeline1.chain(
        DeduplicateStep(subset=["user_id"], keep="first"),
        RenameStep({"UserName": "user_name", "AGE": "age", "Email": "email", "order_amt": "order_amount"}),
        FillNAStep({"age": "median", "order_amount": 0, "email": "unknown@test.com", "city": "未知"}),
        FilterStep(lambda df: df["order_amount"] > 0, "order_amount > 0"),
    )
    result1 = pipeline1.execute(df)
    print(f"\n清洗结果:")
    print(result1.head(10).to_string(index=False))

    # 管道2: 类型转换 + 自定义步骤
    pipeline2 = TransformPipeline("类型转换与增强")
    pipeline2.chain(
        CastStep("age", "int64"),
        CastStep("order_amount", "float64"),
        CustomStep("金额分级", lambda df: df.assign(
            amount_level=pd.cut(df["order_amount"], bins=[0, 100, 500, 10000],
                                labels=["低", "中", "高"])
        )),
        CustomStep("城市标准化", lambda df: df.assign(
            city=df["city"].replace({"": "未知"})
        )),
    )
    result2 = pipeline2.execute(result1)
    print(f"\n转换结果:")
    print(result2.head(10).to_string(index=False))

    # 管道组合: 将两个管道串联
    print("\n" + "=" * 50)
    print("管道组合: 基础清洗 → 类型转换")
    combined = TransformPipeline("组合管道")
    combined.steps = pipeline1.steps + pipeline2.steps
    result_combined = combined.execute(df)
    print(f"\n最终结果: {result_combined.shape}")
    print(result_combined.head(10).to_string(index=False))
    print(f"\n空值统计:\n{result_combined.isna().sum()}")


if __name__ == "__main__":
    main()
