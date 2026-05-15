# 数据来源: seaborn titanic 数据集 (泰坦尼克号乘客数据)
# 综合案例: 完整的特征工程流程, 使用 sklearn Pipeline + ColumnTransformer 组装

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """加载 titanic 数据集"""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except ImportError:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        df = pd.read_csv(url)
    print("=== 原始数据概览 ===")
    print(df.head())
    print(f"\n形状: {df.shape}")
    print(f"\n缺失值:\n{df.isnull().sum()}")
    print(f"\n数据类型:\n{df.dtypes}")
    return df


def step1_imputation(df):
    """步骤1: 缺失值填充"""
    print("\n" + "=" * 60)
    print("步骤1: 缺失值填充")
    print("=" * 60)

    print(f"缺失值统计:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    df_filled = df.copy()

    # 数值列: 中位数填充
    num_cols = df_filled.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_filled[col].isnull().sum() > 0:
            median_val = df_filled[col].median()
            df_filled[col].fillna(median_val, inplace=True)
            print(f"  {col}: 用中位数 {median_val} 填充 {df[col].isnull().sum()} 个缺失值")

    # 分类列: 众数填充
    cat_cols = df_filled.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df_filled[col].isnull().sum() > 0:
            mode_val = df_filled[col].mode()[0]
            df_filled[col].fillna(mode_val, inplace=True)
            print(f"  {col}: 用众数 '{mode_val}' 填充 {df[col].isnull().sum()} 个缺失值")

    # embarked 用众数填充
    if df_filled["embarked"].isnull().sum() > 0:
        mode_emb = df_filled["embarked"].mode()[0]
        df_filled["embarked"].fillna(mode_emb, inplace=True)
        print(f"  embarked: 用众数 '{mode_emb}' 填充")

    print(f"\n填充后缺失值总数: {df_filled.isnull().sum().sum()}")
    return df_filled


def step2_type_conversion(df):
    """步骤2: 类型转换"""
    print("\n" + "=" * 60)
    print("步骤2: 类型转换")
    print("=" * 60)

    df_converted = df.copy()

    # 选择需要保留的列
    keep_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch",
                 "fare", "embarked", "who", "alone"]
    df_converted = df_converted[keep_cols]

    # 分类特征转为字符串类型 (确保 OneHotEncoder 正确处理)
    cat_features = ["pclass", "sex", "embarked", "who", "alone"]
    for col in cat_features:
        df_converted[col] = df_converted[col].astype(str)

    print(f"保留列: {keep_cols}")
    print(f"分类特征: {cat_features}")
    print(f"\n转换后数据类型:\n{df_converted.dtypes}")
    return df_converted


def step3_feature_construction(df):
    """步骤3: 特征构造"""
    print("\n" + "=" * 60)
    print("步骤3: 特征构造")
    print("=" * 60)

    df_feat = df.copy()

    # 家庭规模
    df_feat["family_size"] = df_feat["sibsp"].astype(int) + df_feat["parch"].astype(int) + 1
    # 是否独自一人
    df_feat["is_alone"] = (df_feat["family_size"] == 1).astype(int)
    # 年龄分箱
    df_feat["age_bin"] = pd.cut(
        df_feat["age"].astype(float),
        bins=[0, 12, 18, 35, 60, 100],
        labels=["child", "teen", "young", "middle", "senior"],
    ).astype(str)
    # 票价分箱
    df_feat["fare_bin"] = pd.qcut(
        df_feat["fare"].astype(float),
        q=4,
        labels=["low", "medium", "high", "very_high"],
        duplicates="drop",
    ).astype(str)

    print(f"新增特征: family_size, is_alone, age_bin, fare_bin")
    print(f"\n新特征示例:\n{df_feat[['age', 'age_bin', 'fare', 'fare_bin', 'family_size', 'is_alone']].head(10)}")
    return df_feat


def step4_encoding(df):
    """步骤4: 特征编码"""
    print("\n" + "=" * 60)
    print("步骤4: 特征编码")
    print("=" * 60)

    df_enc = df.copy()

    # 识别数值特征和分类特征
    num_features = df_enc.select_dtypes(include=[np.number]).columns.tolist()
    num_features = [c for c in num_features if c != "survived"]
    cat_features = df_enc.select_dtypes(include=["object"]).columns.tolist()

    print(f"数值特征: {num_features}")
    print(f"分类特征: {cat_features}")

    # OneHotEncoder 编码
    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    cat_data = df_enc[cat_features]
    ohe_result = ohe.fit_transform(cat_data)
    ohe_cols = ohe.get_feature_names_out(cat_features).tolist()

    df_ohe = pd.DataFrame(ohe_result, columns=ohe_cols, index=df_enc.index)
    df_encoded = pd.concat([df_enc[num_features + ["survived"]], df_ohe], axis=1)

    print(f"\n编码后特征数: {len(num_features)} (数值) + {len(ohe_cols)} (独热) = {len(df_encoded.columns) - 1}")
    print(f"编码后列名 (前15): {df_encoded.columns.tolist()[:15]}")
    return df_encoded, num_features, cat_features


def step5_scaling(df_encoded):
    """步骤5: 特征缩放"""
    print("\n" + "=" * 60)
    print("步骤5: 特征缩放")
    print("=" * 60)

    feature_cols = [c for c in df_encoded.columns if c != "survived"]
    target = df_encoded["survived"]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded[feature_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=feature_cols, index=df_encoded.index)
    df_scaled["survived"] = target

    print(f"缩放后统计描述 (前5列):\n{df_scaled[feature_cols[:5]].describe().loc[['mean', 'std']]}")
    return df_scaled


def step6_feature_selection(df_scaled):
    """步骤6: 特征选择"""
    print("\n" + "=" * 60)
    print("步骤6: 特征选择")
    print("=" * 60)

    X = df_scaled.drop("survived", axis=1)
    y = df_scaled["survived"]

    # SelectKBest
    k = min(15, X.shape[1])
    skb = SelectKBest(score_func=f_classif, k=k)
    X_selected = skb.fit_transform(X, y)

    selected_features = skb.get_feature_names_out()
    scores = pd.Series(skb.scores_, index=X.columns).sort_values(ascending=False)

    print(f"SelectKBest (k={k}) 选中特征:")
    for feat in selected_features:
        print(f"  {feat}: F-score={scores[feat]:.2f}")

    print(f"\n所有特征F得分排名 (前15):\n{scores.head(15)}")
    return selected_features


def step7_pipeline(df):
    """步骤7: sklearn Pipeline + ColumnTransformer 组装完整流程"""
    print("\n" + "=" * 60)
    print("步骤7: sklearn Pipeline + ColumnTransformer 完整流程")
    print("=" * 60)

    # 准备数据
    df_pipe = df.copy()
    keep_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch",
                 "fare", "embarked", "who", "alone"]
    df_pipe = df_pipe[keep_cols]

    # 定义特征列
    numeric_features = ["age", "sibsp", "parch", "fare"]
    categorical_features = ["pclass", "sex", "embarked", "who", "alone"]

    # 数值特征 Pipeline: 填充 + 标准化
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 分类特征 Pipeline: 填充 + 独热编码
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    # ColumnTransformer 组合
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # 完整 Pipeline: 预处理 + 特征选择 + 模型
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # 划分数据
    X = df_pipe.drop("survived", axis=1)
    y = df_pipe["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练
    full_pipeline.fit(X_train, y_train)

    # 评估
    y_pred = full_pipeline.predict(X_test)
    print("Pipeline 模型评估 (LogisticRegression):")
    print(classification_report(y_test, y_pred, target_names=["未存活", "存活"]))

    # 查看Pipeline各步骤
    print(f"\nPipeline 步骤:")
    for name, step in full_pipeline.named_steps.items():
        print(f"  {name}: {type(step).__name__}")

    # 查看预处理后的特征名
    ohe = full_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names
    print(f"\n预处理后特征数: {len(all_feature_names)}")
    print(f"特征名 (前10): {all_feature_names[:10]}")

    # 被选中的特征
    selector = full_pipeline.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_names = [all_feature_names[i] for i in range(len(all_feature_names)) if selected_mask[i]]
    print(f"\n特征选择后保留的 {len(selected_names)} 个特征: {selected_names}")

    # 交叉验证
    cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring="accuracy")
    print(f"\n5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 对比: RandomForest Pipeline
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    print(f"\nRandomForest Pipeline 评估:")
    print(classification_report(y_test, y_pred_rf, target_names=["未存活", "存活"]))

    rf_cv = cross_val_score(rf_pipeline, X, y, cv=5, scoring="accuracy")
    print(f"5折交叉验证准确率: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # Pipeline 优势
    print("\nPipeline 优势:")
    print("  1. 防止数据泄露: fit/transform 严格按步骤执行")
    print("  2. 代码简洁: 一次 fit/predict 完成所有步骤")
    print("  3. 可复现: 参数化每一步, 便于调优")
    print("  4. 可部署: 保存 Pipeline 对象即可部署")


def main():
    df = load_data()

    # 分步演示完整流程
    df_filled = step1_imputation(df)
    df_converted = step2_type_conversion(df_filled)
    df_feat = step3_feature_construction(df_converted)
    df_encoded, num_features, cat_features = step4_encoding(df_feat)
    df_scaled = step5_scaling(df_encoded)
    selected_features = step6_feature_selection(df_scaled)

    # Pipeline 组装完整流程
    step7_pipeline(df)

    print("\n" + "=" * 60)
    print("综合案例总结 - 特征工程完整流程:")
    print("  1. 缺失值填充  → SimpleImputer / 手动填充")
    print("  2. 类型转换    → 确保特征类型正确")
    print("  3. 特征构造    → 衍生变量 (家庭规模/分箱/交互)")
    print("  4. 特征编码    → OneHotEncoder 处理分类特征")
    print("  5. 特征缩放    → StandardScaler 标准化")
    print("  6. 特征选择    → SelectKBest 筛选重要特征")
    print("  7. Pipeline组装 → ColumnTransformer + Pipeline 一体化")
    print("=" * 60)


if __name__ == "__main__":
    main()
