import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Trusted AI x RH - Attrition Dashboard",
    layout="wide",
)

# =========================
# PATHS
# =========================
DATA_PATH = "HR_Analytics.csv"
PREDICTIONS_PATH = "attrition_predictions.csv"
METADATA_PATH = "model_metadata.joblib"
DECISION_TREE_PATH = "decision_tree_model.joblib"
AUDIT_FACTORS_PATH = "Audit_Attrition_Complet.xlsx"
HR_KEYS_PATH = "HR_Keys.csv"


# =========================
# LOADERS
# =========================
@st.cache_data
def load_predictions():
    if not Path(PREDICTIONS_PATH).exists():
        raise FileNotFoundError(
            f"Could not find {PREDICTIONS_PATH}. Run: python train_model.py"
        )
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_raw_data():
    if Path(DATA_PATH).exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_metadata():
    if Path(METADATA_PATH).exists():
        return joblib.load(METADATA_PATH)
    return {}


@st.cache_resource
def load_decision_tree():
    if Path(DECISION_TREE_PATH).exists():
        try:
            return joblib.load(DECISION_TREE_PATH)
        except Exception:
            return None
    return None


@st.cache_data
def load_hr_keys():
    if not Path(HR_KEYS_PATH).exists():
        return pd.DataFrame()

    df = pd.read_csv(HR_KEYS_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data
def load_audit_factors():
    if not Path(AUDIT_FACTORS_PATH).exists():
        return pd.DataFrame()

    audit_df = pd.read_excel(AUDIT_FACTORS_PATH)
    audit_df.columns = [str(c).strip() for c in audit_df.columns]

    if "EmpID" not in audit_df.columns:
        return audit_df

    audit_df["EmpID"] = audit_df["EmpID"].astype(str).str.strip()

    if Path(HR_KEYS_PATH).exists():
        keys_df = pd.read_csv(HR_KEYS_PATH)
        keys_df.columns = [str(c).strip() for c in keys_df.columns]

        if "EmpID" in keys_df.columns and "Employee_Hash_ID" in keys_df.columns:
            keys_map = keys_df[["EmpID", "Employee_Hash_ID"]].copy()
            keys_map["EmpID"] = keys_map["EmpID"].astype(str).str.strip()
            keys_map["Employee_Hash_ID"] = keys_map["Employee_Hash_ID"].astype(str).str.strip()

            audit_df = audit_df.merge(
                keys_map,
                on="EmpID",
                how="left",
            )

            # Replace old numeric EmpID with hashed EmpID for matching in the UI
            audit_df["EmpID"] = audit_df["Employee_Hash_ID"].fillna(audit_df["EmpID"]).astype(str).str.strip()

            audit_df = audit_df.drop(columns=["Employee_Hash_ID"], errors="ignore")

    return audit_df

# =========================
# HELPERS
# =========================
def safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)
    return out


def clean_factor_label(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    replacements = {
        "num__TenureYears": "Tenure in years",
        "num__TenureDays": "Tenure in days",
        "num__EmpSatisfaction": "Employee satisfaction",
        "num__EngagementSurvey": "Engagement survey score",
        "num__DaysLateLast30": "Days late in last 30 days",
        "num__Absences": "Absences",
        "num__ManagerID": "Manager group",
        "num__MaritalStatusID": "Marital status",
        "num__PerfScoreID": "Performance score",
        "num__PositionID": "Position",
        "num__DeptID": "Department",
        "num__FromDiversityJobFairID": "Diversity job fair source",
        "num__EmpStatusID": "Employment status",
        "num__Salary": "Salary",
        "num__CitizenDesc": "Citizenship",
        "num__Sex": "Sex",
        "num__RaceDesc": "Race",
        "num__DateofHire": "Hire date",
        "num__LastPerformanceReview_Date": "Last performance review date",
        "num__DaysSinceLastReview": "Days since last review",
        "num__Termd": "Attrition outcome",
    }

    for old, new in replacements.items():
        if text.startswith(old):
            text = text.replace(old, new, 1)

    text = text.replace("num__", "")
    text = text.replace("cat__", "")
    text = text.replace("_", " ")

    return text

def compute_risk_buckets(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["RiskBucket"] = pd.cut(
        temp["PredictedRisk"],
        bins=[-0.001, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )

    candidate_cols = [
        "EmpSatisfaction",
        "EngagementSurvey",
        "DaysLateLast30",
        "Absences",
        "TenureDays",
        "TenureYears",
        "DaysSinceLastReview",
        "PredictedRisk",
    ]
    candidate_cols = [c for c in candidate_cols if c in temp.columns]

    if not candidate_cols:
        return pd.DataFrame()

    grouped = (
        temp.groupby("RiskBucket", dropna=False)[candidate_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    return grouped


def make_security_scan(predictions_df: pd.DataFrame, raw_df: pd.DataFrame, metadata: dict):
    if metadata.get("security_scan"):
        return metadata["security_scan"]

    source_df = raw_df if not raw_df.empty else predictions_df

    sensitive_candidates = [
        "Sex",
        "RaceDesc",
        "CitizenDesc",
        "HispanicLatino",
        "ManagerName",
    ]
    leakage_candidates = [
        "EmpID",
        "Termd",
        "DateofTermination",
        "TermReason",
        "EmploymentStatus",
        "EmpStatusID",
        "DateofHire",
        "LastPerformanceReview_Date",
    ]

    return {
        "rows": int(source_df.shape[0]),
        "columns": int(source_df.shape[1]),
        "target_balance": source_df["Termd"].value_counts(dropna=False).to_dict()
        if "Termd" in source_df.columns
        else {},
        "sensitive_columns_present": [c for c in sensitive_candidates if c in source_df.columns],
        "leakage_columns_present": [c for c in leakage_candidates if c in source_df.columns],
    }


# =========================
# LOAD DATA
# =========================
try:
    predictions_df = load_predictions()
    raw_df = load_raw_data()
    metadata = load_metadata()
    decision_tree_model = load_decision_tree()
    audit_factors_df = load_audit_factors()
    hr_keys_df = load_hr_keys()
except Exception as e:
    st.error(f"Error while loading files: {e}")
    st.stop()

# Keep IDs as strings for safe matching
if "EmpID" in predictions_df.columns:
    predictions_df["EmpID"] = predictions_df["EmpID"].astype(str).str.strip()

if not raw_df.empty and "EmpID" in raw_df.columns:
    raw_df["EmpID"] = raw_df["EmpID"].astype(str).str.strip()

# Optional enrich from raw data
if not raw_df.empty and "EmpID" in predictions_df.columns and "EmpID" in raw_df.columns:
    join_cols = [
        "EmpID",
        "Salary",
        "DateofHire",
        "LastPerformanceReview_Date",
        "Sex",
        "RaceDesc",
        "CitizenDesc",
        "HispanicLatino",
        "ManagerID",
    ]
    join_cols = [c for c in join_cols if c in raw_df.columns]
    raw_subset = raw_df[join_cols].copy()
    predictions_df = predictions_df.merge(raw_subset, on="EmpID", how="left", suffixes=("", "_raw"))

# Create engineered display columns if missing
if "DateofHire" in predictions_df.columns and "TenureDays" not in predictions_df.columns:
    dt = pd.to_datetime(predictions_df["DateofHire"], errors="coerce")
    predictions_df["TenureDays"] = (pd.Timestamp.today() - dt).dt.days

if "DateofHire" in predictions_df.columns and "TenureYears" not in predictions_df.columns:
    dt = pd.to_datetime(predictions_df["DateofHire"], errors="coerce")
    predictions_df["TenureYears"] = ((pd.Timestamp.today() - dt).dt.days / 365.25).round(2)

if "LastPerformanceReview_Date" in predictions_df.columns and "DaysSinceLastReview" not in predictions_df.columns:
    dt = pd.to_datetime(predictions_df["LastPerformanceReview_Date"], errors="coerce")
    predictions_df["DaysSinceLastReview"] = (pd.Timestamp.today() - dt).dt.days


# =========================
# GLOBAL VALUES
# =========================
total_employees = int(len(predictions_df))
actual_attrition_rate = (
    float(predictions_df["Termd"].mean()) if "Termd" in predictions_df.columns else 0.0
)
predicted_high_risk = (
    int((predictions_df["PredictedClass"] == 1).sum())
    if "PredictedClass" in predictions_df.columns
    else 0
)
roc_auc = float(metadata.get("roc_auc", metadata.get("metrics", {}).get("roc_auc", 0.0)))
model_name = metadata.get("model_name", "Logistic Regression")

risk_bucket_table = compute_risk_buckets(predictions_df)
fairness_df = pd.DataFrame(metadata.get("fairness_audit", []))
security_scan = make_security_scan(predictions_df, raw_df, metadata)


# =========================
# HEADER
# =========================
st.title("Trusted AI x RH - Attrition Dashboard")
st.caption("Hackathon demo focused on explainability, cybersecurity, fairness and frugality.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Employees", total_employees)
m2.metric("Actual attrition rate", f"{actual_attrition_rate * 100:.1f}%")
m3.metric("Predicted high-risk employees", predicted_high_risk)
m4.metric("Model ROC-AUC", f"{roc_auc:.3f}")

tabs = st.tabs(
    [
        "Business analysis",
        "Single employee",
        "High-risk table",
        "Fairness audit",
        "Security + model card",
    ]
)
# =========================
# TAB 1 - OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Project summary")

    st.write(
        "This dashboard supports HR attrition analysis by combining dataset exploration, "
        "risk prediction, fairness monitoring, cybersecurity checks, and employee-level review."
    )

    st.markdown("### Business objective")
    st.write(
        "The goal is to help HR identify attrition patterns early, understand which employee profiles "
        "are more exposed to turnover risk, and support preventive action without making automatic HR decisions."
    )

    st.markdown("### Dataset overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Employees", total_employees)
    k2.metric("Actual attrition rate", f"{actual_attrition_rate * 100:.1f}%")

    avg_engagement = float(predictions_df["EngagementSurvey"].mean()) if "EngagementSurvey" in predictions_df.columns else 0.0
    avg_satisfaction = float(predictions_df["EmpSatisfaction"].mean()) if "EmpSatisfaction" in predictions_df.columns else 0.0

    k3.metric("Avg engagement", f"{avg_engagement:.2f}")
    k4.metric("Avg satisfaction", f"{avg_satisfaction:.2f}")

    st.markdown("### Average profile: attrition vs no attrition")

    compare_cols = [
        "EngagementSurvey",
        "EmpSatisfaction",
        "DaysLateLast30",
        "Absences",
        "TenureDays",
        "TenureYears",
        "DaysSinceLastReview",
        "Salary",
    ]
    compare_cols = [c for c in compare_cols if c in predictions_df.columns]

    if "Termd" in predictions_df.columns and compare_cols:
        attrition_profile = (
            predictions_df.groupby("Termd")[compare_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

        attrition_profile["Termd"] = attrition_profile["Termd"].map(
            {0: "Stayed", 1: "Left"}
        )

        numeric_cols = attrition_profile.select_dtypes(include=[np.number]).columns
        attrition_profile[numeric_cols] = attrition_profile[numeric_cols].round(2)

        st.dataframe(attrition_profile, use_container_width=True)
    else:
        st.info("Attrition profile comparison is not available.")

    st.markdown("### Attrition by department")

    if "Department" in predictions_df.columns and "Termd" in predictions_df.columns:
        dept_summary = (
            predictions_df.groupby("Department", dropna=False)
            .agg(
                employee_count=("EmpID", "count"),
                actual_attrition_rate=("Termd", "mean"),
            )
            .reset_index()
        )

        dept_summary["actual_attrition_rate"] = (
            dept_summary["actual_attrition_rate"] * 100
        ).round(2)

        dept_summary = dept_summary.sort_values(
            by="actual_attrition_rate", ascending=False
        ).reset_index(drop=True)

        st.dataframe(dept_summary, use_container_width=True)
    else:
        st.info("Department attrition summary is not available.")

    st.markdown("### Predicted risk bucket summary")

    if not risk_bucket_table.empty:
        display_bucket = risk_bucket_table.copy()
        numeric_cols = display_bucket.select_dtypes(include=[np.number]).columns
        display_bucket[numeric_cols] = display_bucket[numeric_cols].round(3)
        st.dataframe(display_bucket, use_container_width=True)
    else:
        st.info("Risk bucket summary is not available.")

    st.markdown("### Modeling note")
    st.info(
        "Detailed experimentation and analysis were performed in the project notebook. "
        "This dashboard focuses on the final business view and operational interpretation."
    )

# =========================
# TAB 2 - SINGLE EMPLOYEE
# =========================
with tabs[1]:
    st.subheader("Explain one employee")

    emp_options = sorted(predictions_df["EmpID"].dropna().astype(str).unique().tolist())
    selected_emp_id = st.selectbox("Choose an employee", emp_options)

    selected_row = predictions_df[predictions_df["EmpID"].astype(str).str.strip() == str(selected_emp_id).strip()].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted risk", f"{float(selected_row['PredictedRisk']) * 100:.2f}%")
    with c2:
        st.metric("Predicted class", int(selected_row["PredictedClass"]))
    with c3:
        actual_value = (
            int(selected_row["Termd"])
            if "Termd" in selected_row.index and pd.notna(selected_row["Termd"])
            else "N/A"
        )
        st.metric("Actual class", actual_value)

    snapshot_cols = [
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
        "TenureYears",
        "DaysSinceLastReview",
    ]
    snapshot_cols = [c for c in snapshot_cols if c in selected_row.index]

    st.write("Employee snapshot:")
    snapshot_df = pd.DataFrame(
        {"value": [selected_row[c] for c in snapshot_cols]},
        index=snapshot_cols,
    )
    st.dataframe(safe_display_df(snapshot_df), use_container_width=True)

    st.info("SHAP explanation is not available in this environment.")

    if decision_tree_model is not None:
        st.write("Decision tree backup model detected.")
    else:
        st.caption("Decision tree backup model not loaded in this environment.")

    st.markdown("### Factors from audit file")

    selected_emp_id_str = str(selected_emp_id).strip()

    employee_factors = pd.DataFrame()
    if not audit_factors_df.empty and "EmpID" in audit_factors_df.columns:
        employee_factors = audit_factors_df[
            audit_factors_df["EmpID"].astype(str).str.strip() == selected_emp_id_str
        ]

    if not employee_factors.empty:
        factor_row = employee_factors.iloc[0]

        risque_excel = factor_row["Risque_%"] if "Risque_%" in factor_row.index else None
        if pd.notna(risque_excel):
            st.write(f"**Risk from audit file:** {risque_excel}%")

        facteur_cols = [
            c for c in ["Facteur_1", "Facteur_2", "Facteur_3", "Facteur_4", "Facteur_5"]
            if c in factor_row.index
        ]

        displayed_factors = []
        for c in facteur_cols:
            value = factor_row[c]
            if pd.notna(value) and str(value).strip() != "":
                displayed_factors.append(clean_factor_label(value))

        if displayed_factors:
            factors_table = pd.DataFrame(
                {
                    "Factor": [f"Facteur {i}" for i in range(1, len(displayed_factors) + 1)],
                    "Value": displayed_factors,
                }
            )
            st.dataframe(safe_display_df(factors_table), use_container_width=True)
        else:
            st.warning("No factors found for this employee in the audit file.")
    else:
        st.warning("This employee is not present in Audit_Attrition_Complet.xlsx / HR_Keys.csv mapping.")

# =========================
# TAB 3 - HIGH-RISK TABLE
# =========================
with tabs[2]:
    st.subheader("Employees sorted by attrition risk")

    display_cols = [
        "EmpID",
        "ManagerID",
        "Department",
        "Position",
        "PerformanceScore",
        "EngagementSurvey",
        "EmpSatisfaction",
        "DaysLateLast30",
        "Absences",
        "TenureDays",
        "TenureYears",
        "DaysSinceLastReview",
        "PredictedRisk",
        "PredictedClass",
        "Termd",
    ]
    display_cols = [c for c in display_cols if c in predictions_df.columns]

    high_risk_df = predictions_df[display_cols].copy()

    st.markdown("### Filters")

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        manager_options = []
        if "ManagerID" in high_risk_df.columns:
            manager_options = sorted(
                [x for x in high_risk_df["ManagerID"].dropna().unique().tolist()]
            )
        selected_managers = st.multiselect(
            "Filter by ManagerID",
            options=manager_options,
            default=[],
        )

    with f2:
        department_options = []
        if "Department" in high_risk_df.columns:
            department_options = sorted(
                [str(x) for x in high_risk_df["Department"].dropna().unique().tolist()]
            )
        selected_departments = st.multiselect(
            "Filter by Department",
            options=department_options,
            default=[],
        )

    with f3:
        position_options = []
        if "Position" in high_risk_df.columns:
            position_options = sorted(
                [str(x) for x in high_risk_df["Position"].dropna().unique().tolist()]
            )
        selected_positions = st.multiselect(
            "Filter by Position",
            options=position_options,
            default=[],
        )

    with f4:
        class_options = []
        if "PredictedClass" in high_risk_df.columns:
            class_options = sorted(
                [int(x) for x in high_risk_df["PredictedClass"].dropna().unique().tolist()]
            )
        selected_classes = st.multiselect(
            "Filter by Predicted class",
            options=class_options,
            default=[],
        )

    g1, g2 = st.columns(2)

    with g1:
        min_risk = st.slider(
            "Minimum predicted risk (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )

    with g2:
        only_actual_attrition = st.checkbox("Show only actual attrition = 1", value=False)

    filtered_df = high_risk_df.copy()

    if selected_managers and "ManagerID" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["ManagerID"].isin(selected_managers)]

    if selected_departments and "Department" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Department"].astype(str).isin(selected_departments)]

    if selected_positions and "Position" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Position"].astype(str).isin(selected_positions)]

    if selected_classes and "PredictedClass" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["PredictedClass"].isin(selected_classes)]

    if "PredictedRisk" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["PredictedRisk"] * 100 >= min_risk]

    if only_actual_attrition and "Termd" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Termd"] == 1]

    filtered_df = filtered_df.sort_values(by="PredictedRisk", ascending=False).reset_index(drop=True)

    if "ManagerID" in filtered_df.columns:
        st.markdown("### Manager summary")

        manager_summary = (
            filtered_df.groupby(["ManagerID"], dropna=False)
            .agg(
                employee_count=("EmpID", "count"),
                avg_predicted_risk=("PredictedRisk", "mean"),
                predicted_high_risk_count=("PredictedClass", "sum"),
                actual_attrition_count=("Termd", "sum"),
            )
            .reset_index()
        )

        manager_summary["avg_predicted_risk"] = (manager_summary["avg_predicted_risk"] * 100).round(2)
        manager_summary = manager_summary.sort_values(
            by=["avg_predicted_risk", "predicted_high_risk_count"],
            ascending=[False, False],
        ).reset_index(drop=True)

        st.dataframe(manager_summary, use_container_width=True)

    st.markdown("### Filtered employee table")

    display_table = filtered_df.copy()
    if "PredictedRisk" in display_table.columns:
        display_table["PredictedRisk"] = (display_table["PredictedRisk"] * 100).round(2).astype(str) + "%"

    st.dataframe(display_table, use_container_width=True)

# =========================
# TAB 4 - FAIRNESS AUDIT
# =========================
with tabs[3]:
    st.subheader("Fairness audit")
    st.write(
        "Sensitive attributes are removed from training, but we still audit outcomes by subgroup "
        "to check whether the model behaves very differently across groups."
    )

    if not fairness_df.empty:
        fairness_display = fairness_df.copy()
        numeric_cols = fairness_display.select_dtypes(include=[np.number]).columns
        fairness_display[numeric_cols] = fairness_display[numeric_cols].round(4)
        st.dataframe(fairness_display, use_container_width=True)
    else:
        fallback_rows = []
        for attr in ["Sex", "RaceDesc"]:
            if attr in predictions_df.columns and "PredictedRisk" in predictions_df.columns:
                for group_value, g in predictions_df.groupby(attr, dropna=False):
                    fallback_rows.append(
                        {
                            "attribute": attr,
                            "group": str(group_value),
                            "count": int(len(g)),
                            "actual_attrition_rate": float(g["Termd"].mean()) if "Termd" in g.columns else np.nan,
                            "predicted_attrition_rate": float(g["PredictedClass"].mean()) if "PredictedClass" in g.columns else np.nan,
                            "avg_risk_score": float(g["PredictedRisk"].mean()),
                        }
                    )
        fallback_df = pd.DataFrame(fallback_rows)
        if not fallback_df.empty:
            st.dataframe(fallback_df.round(4), use_container_width=True)
        else:
            st.info("Fairness audit table is not available.")

# =========================
# TAB 5 - SECURITY + MODEL CARD
# =========================
with tabs[4]:
    st.subheader("Cybersecurity + responsible AI checklist")

    st.markdown("**Security scan:**")
    st.json(security_scan)

    st.markdown("**Intended use**")
    st.write(
        "This tool is designed for HR decision support. It helps prioritize employees for preventive "
        "review, not for automatic sanction, hiring, or firing decisions."
    )

    st.markdown("**Responsible AI choices**")
    st.write(
        "- Sensitive attributes are excluded from model training.\n"
        "- Leakage columns are removed so the model does not cheat.\n"
        "- Fairness is audited by subgroup after prediction.\n"
        "- The final system keeps a transparent dashboard for human review.\n"
        "- A classical ML comparison pipeline is used instead of unnecessary heavy models."
    )

    st.markdown("**Model card summary**")
    model_card = {
        "selected_model": model_name,
        "dataset_rows": total_employees,
        "actual_attrition_rate": round(actual_attrition_rate, 4),
        "predicted_high_risk_employees": predicted_high_risk,
        "roc_auc": round(roc_auc, 4),
        "training_columns": metadata.get("training_columns", []),
        "categorical_columns": metadata.get("categorical_columns", []),
        "numerical_columns": metadata.get("numerical_columns", []),
        "limitations": [
            "Dataset may be synthetic and cleaner than real enterprise data.",
            "Fairness monitoring is indicative and should be extended with larger real-world samples.",
            "The model identifies predictive patterns, not proven causal reasons.",
            "Human oversight remains necessary for any HR action.",
        ],
    }
    st.json(model_card)