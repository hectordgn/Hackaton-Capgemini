from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Project constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HR_Analytics.csv"
MODEL_PATH = BASE_DIR / "attrition_model.joblib"
TREE_PATH = BASE_DIR / "attrition_tree.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.joblib"
RISK_TABLE_PATH = BASE_DIR / "high_risk_employees.csv"
TREE_RULES_PATH = BASE_DIR / "decision_tree_rules.txt"

TARGET_COL = "Termd"

SENSITIVE_COLUMNS = [
    "Sex",
    "RaceDesc",
    "CitizenDesc",
    "HispanicLatino",
    "ManagerName",
]

LEAKAGE_COLUMNS = [
    "EmpID",
    "Termd",
    "DateofTermination",
    "TermReason",
    "EmploymentStatus",
    "EmpStatusID",
    "DateofHire",
    "LastPerformanceReview_Date",
]

# We keep the final feature set focused, frugal, and easy to explain.
MODEL_FEATURES = [
    "Salary",
    "Department",
    "Position",
    "State",
    "RecruitmentSource",
    "PerformanceScore",
    "EngagementSurvey",
    "EmpSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "TenureDays",
    "DaysSinceLastReview",
]

DISPLAY_COLUMNS = [
    "EmpID",
    "Department",
    "Position",
    "Salary",
    "PerformanceScore",
    "EngagementSurvey",
    "EmpSatisfaction",
    "DaysLateLast30",
    "Absences",
    "TenureDays",
    "DaysSinceLastReview",
]


# -----------------------------
# Data loading / scanning
# -----------------------------
def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {path}")

    df = pd.read_csv(path)

    required = {
        "DateofHire",
        "LastPerformanceReview_Date",
        TARGET_COL,
        "EmpID",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df



def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    work_df["DateofHire"] = pd.to_datetime(work_df["DateofHire"], errors="coerce")
    work_df["LastPerformanceReview_Date"] = pd.to_datetime(
        work_df["LastPerformanceReview_Date"], errors="coerce"
    )

    reference_date = pd.Timestamp.today().normalize()
    work_df["TenureDays"] = (reference_date - work_df["DateofHire"]).dt.days
    work_df["DaysSinceLastReview"] = (
        reference_date - work_df["LastPerformanceReview_Date"]
    ).dt.days

    return work_df



def security_scan(df: pd.DataFrame) -> Dict[str, object]:
    null_counts = df.isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    scan = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_balance": df[TARGET_COL].value_counts(dropna=False).to_dict(),
        "sensitive_columns_present": [c for c in SENSITIVE_COLUMNS if c in df.columns],
        "leakage_columns_present": [c for c in LEAKAGE_COLUMNS if c in df.columns],
        "null_columns": null_counts.to_dict(),
    }
    return scan



def anonymize_preview(df: pd.DataFrame) -> pd.DataFrame:
    preview = df.copy()
    if "ManagerName" in preview.columns:
        preview["ManagerName"] = "[REDACTED_MANAGER]"
    if "EmpID" in preview.columns:
        preview["EmpID"] = [f"EMP_{i:03d}" for i in range(1, len(preview) + 1)]
    return preview


# -----------------------------
# Model preparation
# -----------------------------
def build_model_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    work_df = add_derived_features(df)

    missing_features = [c for c in MODEL_FEATURES if c not in work_df.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    X = work_df[MODEL_FEATURES].copy()
    y = work_df[TARGET_COL].copy()
    return X, y, work_df



def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]
    return numeric_features, categorical_features



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


# -----------------------------
# Explanation helpers
# -----------------------------
def get_prediction_columns(full_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLUMNS if c in full_df.columns]
    return full_df[cols].copy()



def simplify_feature_name(feature_name: str) -> str:
    feature_name = feature_name.replace("num__", "")
    feature_name = feature_name.replace("cat__", "")
    return feature_name



def explain_row_with_linear_shap(bundle: Dict[str, object], row: pd.DataFrame, top_k: int = 5):
    try:
        import shap  # noqa: WPS433

        pipeline = bundle["pipeline"]
        X_background = bundle["X_train_sample"]

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        transformed_background = preprocessor.transform(X_background)
        transformed_row = preprocessor.transform(row)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.LinearExplainer(
            model,
            transformed_background,
            feature_names=feature_names,
        )
        shap_values = explainer(transformed_row)
        values = shap_values.values[0]

        pairs = []
        for name, value in zip(feature_names, values):
            pairs.append(
                {
                    "feature": simplify_feature_name(name),
                    "impact": float(value),
                    "direction": "increases risk" if value > 0 else "reduces risk",
                }
            )

        pairs = sorted(pairs, key=lambda x: abs(x["impact"]), reverse=True)[:top_k]
        return pairs
    except Exception:
        return []



def bundle_summary_json(scan: Dict[str, object]) -> str:
    return json.dumps(scan, indent=2, default=str)
