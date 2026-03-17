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
    df = pd.read_csv(PREDICTIONS_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data
def load_raw_data():
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH)
        df.columns = [str(c).strip() for c in df.columns]
        return df
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
    return audit_df


# =========================
# HELPERS
# =========================
def normalize_id_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )


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


def risk_bucket_from_score(score):
    if pd.isna(score):
        return np.nan
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"


def compute_risk_buckets(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    if "DisplayRisk" not in temp.columns:
        return pd.DataFrame()

    temp["RiskBucket"] = pd.cut(
        temp["DisplayRisk"],
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
        "DisplayRisk",
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

# =========================
# NORMALIZE IDs
# =========================
if "EmpID" in predictions_df.columns:
    predictions_df["EmpID"] = normalize_id_series(predictions_df["EmpID"])

if not raw_df.empty and "EmpID" in raw_df.columns:
    raw_df["EmpID"] = normalize_id_series(raw_df["EmpID"])

if not audit_factors_df.empty and "EmpID" in audit_factors_df.columns:
    audit_factors_df["EmpID"] = normalize_id_series(audit_factors_df["EmpID"])

if not hr_keys_df.empty and "EmpID" in hr_keys_df.columns:
    hr_keys_df["EmpID"] = normalize_id_series(hr_keys_df["EmpID"])

# =========================
# OPTIONAL RAW DATA ENRICHMENT
# =========================
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
        "Department",
        "Position",
        "PerformanceScore",
        "EngagementSurvey",
        "EmpSatisfaction",
        "DaysLateLast30",
        "Absences",
        "Termd",
    ]
    join_cols = [c for c in join_cols if c in raw_df.columns]
    raw_subset = raw_df[join_cols].copy()
    predictions_df = predictions_df.merge(
        raw_subset,
        on="EmpID",
        how="left",
        suffixes=("", "_raw"),
    )

# =========================
# ENGINEERED DISPLAY COLUMNS
# =========================
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
# MATCH AUDIT FILE TO SAME EMPLOYEE TABLE
# =========================
if not audit_factors_df.empty and "EmpID" in audit_factors_df.columns:
    audit_merge_df = audit_factors_df.copy()

    possible_hash_cols = []
    if not hr_keys_df.empty:
        possible_hash_cols = [
            c for c in hr_keys_df.columns
            if c.lower() in ["employee_hash_id", "hash_id", "hashed_empid"]
        ]

    if not hr_keys_df.empty and possible_hash_cols:
        hash_col = possible_hash_cols[0]

        keys_map = hr_keys_df[["EmpID", hash_col]].copy()
        keys_map["EmpID"] = normalize_id_series(keys_map["EmpID"])
        keys_map[hash_col] = normalize_id_series(keys_map[hash_col])

        audit_merge_df = audit_merge_df.merge(keys_map, on="EmpID", how="left")
        audit_merge_df["AuditMatchID"] = audit_merge_df[hash_col].fillna(audit_merge_df["EmpID"])
    else:
        audit_merge_df["AuditMatchID"] = audit_merge_df["EmpID"]

    keep_audit_cols = [
        "AuditMatchID",
        "Risque_%",
        "Facteur_1",
        "Facteur_2",
        "Facteur_3",
        "Facteur_4",
        "Facteur_5",
    ]
    keep_audit_cols = [c for c in keep_audit_cols if c in audit_merge_df.columns]
    audit_merge_df = audit_merge_df[keep_audit_cols].copy()

    if "Risque_%" in audit_merge_df.columns:
        audit_merge_df["AuditRisk"] = pd.to_numeric(audit_merge_df["Risque_%"], errors="coerce") / 100.0
        audit_merge_df["AuditClass"] = (audit_merge_df["AuditRisk"] >= 0.5).astype("Int64")
        audit_merge_df["AuditBucket"] = audit_merge_df["AuditRisk"].apply(risk_bucket_from_score)

    predictions_df["PredMatchID"] = predictions_df["EmpID"]

    predictions_df = predictions_df.merge(
        audit_merge_df,
        left_on="PredMatchID",
        right_on="AuditMatchID",
        how="left",
    )

# =========================
# UNIFIED DISPLAY COLUMNS
# =========================
if "AuditRisk" in predictions_df.columns:
    predictions_df["DisplayRisk"] = predictions_df["AuditRisk"].fillna(predictions_df["PredictedRisk"])
else:
    predictions_df["DisplayRisk"] = predictions_df["PredictedRisk"]

if "AuditClass" in predictions_df.columns:
    predictions_df["DisplayClass"] = predictions_df["AuditClass"].fillna(predictions_df["PredictedClass"])
else:
    predictions_df["DisplayClass"] = predictions_df["PredictedClass"]

predictions_df["DisplayClass"] = pd.to_numeric(
    predictions_df["DisplayClass"], errors="coerce"
).fillna(0).astype(int)

predictions_df["RiskBucket"] = predictions_df["DisplayRisk"].apply(risk_bucket_from_score)

# =========================
# UNIFIED UI EMPLOYEE ID
# =========================
ui_emp_id_col = "EmpID"

if not hr_keys_df.empty and "EmpID" in hr_keys_df.columns:
    possible_hash_cols = [
        c for c in hr_keys_df.columns
        if c.lower() in ["employee_hash_id", "hash_id", "hashed_empid"]
    ]

    if possible_hash_cols:
        hash_col = possible_hash_cols[0]

        keys_ui = hr_keys_df[["EmpID", hash_col]].copy()
        keys_ui["EmpID"] = normalize_id_series(keys_ui["EmpID"])
        keys_ui[hash_col] = normalize_id_series(keys_ui[hash_col])

        predictions_df = predictions_df.merge(
            keys_ui,
            on="EmpID",
            how="left",
            suffixes=("", "_ui"),
        )

        predictions_df["UIEmpID"] = predictions_df[hash_col].fillna(predictions_df["EmpID"])
        ui_emp_id_col = "UIEmpID"
    else:
        predictions_df["UIEmpID"] = predictions_df["EmpID"]
        ui_emp_id_col = "UIEmpID"
else:
    predictions_df["UIEmpID"] = predictions_df["EmpID"]
    ui_emp_id_col = "UIEmpID"

predictions_df["UIEmpID"] = normalize_id_series(predictions_df["UIEmpID"])

# =========================
# GLOBAL VALUES
# =========================
# =========================
# GLOBAL VALUES
# =========================
total_employees = int(len(predictions_df))
actual_attrition_rate = (
    float(predictions_df["Termd"].mean()) if "Termd" in predictions_df.columns else 0.0
)
model_name = metadata.get("model_name", "Logistic Regression")
risk_bucket_table = compute_risk_buckets(predictions_df)
fairness_df = pd.DataFrame(metadata.get("fairness_audit", []))
security_scan = make_security_scan(predictions_df, raw_df, metadata)
audit_linked = "Yes" if "AuditRisk" in predictions_df.columns and predictions_df["AuditRisk"].notna().any() else "No"

# use audit count first for top metric
if "AuditClass" in predictions_df.columns and predictions_df["AuditClass"].notna().any():
    predicted_high_risk = int((predictions_df["AuditClass"] == 1).sum())
else:
    predicted_high_risk = (
        int((predictions_df["DisplayClass"] == 1).sum())
        if "DisplayClass" in predictions_df.columns
        else 0
    )
# =========================
# HEADER
# =========================
st.title("Trusted AI x RH - Attrition Dashboard")
st.caption("Hackathon demo focused on explainability, cybersecurity, fairness and frugality.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Employees", total_employees)
m2.metric("Actual attrition rate", f"{actual_attrition_rate * 100:.1f}%")
m3.metric("Predicted high-risk employees", predicted_high_risk)
m4.metric("Audit file linked", audit_linked)

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
                employee_count=(ui_emp_id_col, "count"),
                actual_attrition_rate=("Termd", "mean"),
                avg_display_risk=("DisplayRisk", "mean"),
            )
            .reset_index()
        )

        dept_summary["actual_attrition_rate"] = (
            dept_summary["actual_attrition_rate"] * 100
        ).round(2)
        dept_summary["avg_display_risk"] = (
            dept_summary["avg_display_risk"] * 100
        ).round(2)

        dept_summary = dept_summary.sort_values(
            by="avg_display_risk", ascending=False
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

    emp_options = sorted(
        predictions_df["UIEmpID"].dropna().astype(str).unique().tolist()
    )
    selected_emp_id = st.selectbox("Choose an employee", emp_options)

    selected_row = predictions_df[
        predictions_df["UIEmpID"].astype(str).str.strip() == str(selected_emp_id).strip()
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted risk", f"{float(selected_row['DisplayRisk']) * 100:.2f}%")
    with c2:
        st.metric("Predicted class", int(selected_row["DisplayClass"]))
    with c3:
        actual_value = (
            int(selected_row["Termd"])
            if "Termd" in selected_row.index and pd.notna(selected_row["Termd"])
            else "N/A"
        )
        st.metric("Actual class", actual_value)

    snapshot_cols = [
        "UIEmpID",
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

    if "AuditRisk" in selected_row.index and pd.notna(selected_row["AuditRisk"]):
        st.write(f"**Risk from audit file:** {float(selected_row['AuditRisk']) * 100:.2f}%")

        facteur_cols = [
            c for c in ["Facteur_1", "Facteur_2", "Facteur_3", "Facteur_4", "Facteur_5"]
            if c in selected_row.index
        ]

        displayed_factors = []
        for c in facteur_cols:
            value = selected_row[c]
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
    st.subheader("Employees sorted by attrition risk (audit file)")

    # ONLY use rows that exist in the audit file
    if "AuditRisk" not in predictions_df.columns or predictions_df["AuditRisk"].notna().sum() == 0:
        st.warning("No audit-file risk values were found.")
    else:
        audit_table_df = predictions_df[predictions_df["AuditRisk"].notna()].copy()

        # use audit risk/class/bucket only
        audit_table_df["AuditRiskBucket"] = audit_table_df["AuditRisk"].apply(risk_bucket_from_score)
        audit_table_df["AuditClassDisplay"] = pd.to_numeric(
            audit_table_df["AuditClass"], errors="coerce"
        ).fillna(0).astype(int)

        st.markdown("### Filters")

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            manager_options = []
            if "ManagerID" in audit_table_df.columns:
                manager_options = sorted(
                    [x for x in audit_table_df["ManagerID"].dropna().unique().tolist()]
                )
            selected_managers = st.multiselect(
                "Filter by ManagerID",
                options=manager_options,
                default=[],
            )

        with f2:
            department_options = []
            if "Department" in audit_table_df.columns:
                department_options = sorted(
                    [str(x) for x in audit_table_df["Department"].dropna().unique().tolist()]
                )
            selected_departments = st.multiselect(
                "Filter by Department",
                options=department_options,
                default=[],
            )

        with f3:
            position_options = []
            if "Position" in audit_table_df.columns:
                position_options = sorted(
                    [str(x) for x in audit_table_df["Position"].dropna().unique().tolist()]
                )
            selected_positions = st.multiselect(
                "Filter by Position",
                options=position_options,
                default=[],
            )

        with f4:
            class_options = sorted(
                [int(x) for x in audit_table_df["AuditClassDisplay"].dropna().unique().tolist()]
            )
            selected_classes = st.multiselect(
                "Filter by Audit class",
                options=class_options,
                default=[],
            )

        g1, g2 = st.columns(2)

        with g1:
            min_risk = st.slider(
                "Minimum audit risk (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
            )

        with g2:
            only_actual_attrition = st.checkbox("Show only actual attrition = 1", value=False)

        filtered_df = audit_table_df.copy()

        if selected_managers and "ManagerID" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["ManagerID"].isin(selected_managers)]

        if selected_departments and "Department" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Department"].astype(str).isin(selected_departments)]

        if selected_positions and "Position" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Position"].astype(str).isin(selected_positions)]

        if selected_classes:
            filtered_df = filtered_df[filtered_df["AuditClassDisplay"].isin(selected_classes)]

        filtered_df = filtered_df[filtered_df["AuditRisk"] * 100 >= min_risk]

        if only_actual_attrition and "Termd" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Termd"] == 1]

        filtered_df = filtered_df.sort_values(by="AuditRisk", ascending=False).reset_index(drop=True)

        if "ManagerID" in filtered_df.columns:
            st.markdown("### Manager summary (audit file)")

            manager_summary = (
                filtered_df.groupby(["ManagerID"], dropna=False)
                .agg(
                    employee_count=("UIEmpID", "count"),
                    avg_audit_risk=("AuditRisk", "mean"),
                    audit_high_risk_count=("AuditClassDisplay", "sum"),
                    actual_attrition_count=("Termd", "sum"),
                )
                .reset_index()
            )

            manager_summary["avg_audit_risk"] = (manager_summary["avg_audit_risk"] * 100).round(2)
            manager_summary = manager_summary.sort_values(
                by=["avg_audit_risk", "audit_high_risk_count"],
                ascending=[False, False],
            ).reset_index(drop=True)

            st.dataframe(manager_summary, use_container_width=True)

        st.markdown("### Filtered employee table")

        show_cols = [
            "UIEmpID",
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
            "AuditRisk",
            "AuditClassDisplay",
            "Termd",
            "AuditRiskBucket",
        ]
        show_cols = [c for c in show_cols if c in filtered_df.columns]

        display_table = filtered_df[show_cols].copy()
        if "AuditRisk" in display_table.columns:
            display_table["AuditRisk"] = (display_table["AuditRisk"] * 100).round(2).astype(str) + "%"

        display_table = display_table.rename(
            columns={
                "UIEmpID": "EmpID",
                "AuditRisk": "RiskFromAudit",
                "AuditClassDisplay": "AuditClass",
                "AuditRiskBucket": "RiskBucket",
            }
        )

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
            if attr in predictions_df.columns and "DisplayRisk" in predictions_df.columns:
                for group_value, g in predictions_df.groupby(attr, dropna=False):
                    fallback_rows.append(
                        {
                            "attribute": attr,
                            "group": str(group_value),
                            "count": int(len(g)),
                            "actual_attrition_rate": float(g["Termd"].mean()) if "Termd" in g.columns else np.nan,
                            "predicted_attrition_rate": float(g["DisplayClass"].mean()) if "DisplayClass" in g.columns else np.nan,
                            "avg_risk_score": float(g["DisplayRisk"].mean()),
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
        "audit_file_linked": audit_linked,
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