"""
CKD Risk Prediction — Streamlit Web Dashboard
Interactive web application for CKD risk assessment and model exploration.

Dataset: final_verified_ckd_dataset_fixed.xlsx (2159 patients)
Split strategy: 70% Train / 20% Validation / 10% Test
eGFR, serum_creatinine excluded from prediction (used for staging only).
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
from pathlib import Path

from src.data_loader import load_data, get_display_name, get_dataset_summary
from src.preprocessing import (
    encode_features, NUMERICAL_FEATURES, BINARY_YES_NO_COLUMNS,
    EXCLUDE_FROM_PREDICTION,
)
from src.risk_scoring import get_risk_category, get_risk_color
from src.staging import classify_egfr_stage, get_stage_description, get_stage_color
from src.explainability import explain_patient

# ── Page Configuration ──
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Disclaimer ──
st.markdown(
    """
    <div style="background-color:#fff3cd; padding:10px; border-radius:5px;
    border-left:5px solid #ffc107; margin-bottom:15px;">
    <strong>Disclaimer:</strong> This tool is for decision support only.
    Not a replacement for medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Cached Data & Model Loaders ──
@st.cache_data
def load_dataset():
    """Load and cache the dataset."""
    return load_data("data/final_verified_ckd_dataset_fixed.xlsx")


@st.cache_data
def load_encoded_dataset():
    """Load and cache encoded dataset."""
    df = load_data("data/final_verified_ckd_dataset_fixed.xlsx")
    return encode_features(df)


@st.cache_resource
def load_trained_models():
    """Load all trained models from disk."""
    from src.model_training import load_models
    return load_models("models")


@st.cache_data
def load_comparison_table():
    """Load model comparison CSV."""
    path = Path("outputs/model_comparison.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_test_comparison_table():
    """Load test set model comparison CSV."""
    path = Path("outputs/model_comparison_test.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_risk_scores():
    """Load risk scores CSV."""
    path = Path("outputs/risk_scores.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_staging_data():
    """Load CKD staging CSV."""
    path = Path("outputs/ckd_staging.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_feature_names():
    """Load feature names."""
    path = Path("models/feature_names.joblib")
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def compute_shap_cache(model_name):
    """Compute and cache SHAP values using validation set.
    Falls back to best base model if Stacking is selected (TreeExplainer limitation).
    """
    models = load_trained_models()
    model = models.get(model_name)
    feature_names = load_feature_names()

    if model is None or feature_names is None:
        return None, None, None

    # TreeExplainer doesn't support StackingClassifier — use best base model
    if model_name == "Stacking":
        comp_df = load_comparison_table()
        if comp_df is not None:
            base_df = comp_df[comp_df["Model"] != "Stacking"]
            base_sorted = base_df.sort_values(by=["Recall", "F1-Score"], ascending=False)
            fallback_name = base_sorted.iloc[0]["Model"]
            model = models.get(fallback_name, model)

    X_val_sc, y_val, _, _ = get_validation_data()

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val_sc.values)
        if isinstance(sv, list):
            sv = sv[1]
        return sv, X_val_sc.values, y_val
    except Exception:
        return None, None, None


def get_validation_data():
    """Get preprocessed validation data and indices (mirrors 70/20/10 split)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = load_data("data/final_verified_ckd_dataset_fixed.xlsx")
    df_enc = encode_features(df)

    # Exclude eGFR from prediction features
    feature_cols = [
        c for c in df_enc.columns
        if c != "ckd_diagnosis" and c not in EXCLUDE_FROM_PREDICTION
    ]
    X = df_enc[feature_cols]
    y = df_enc["ckd_diagnosis"]

    # Mirror the 70/20/10 split
    X_temp, X_test, y_temp, y_test, temp_idx, test_idx = train_test_split(
        X, y, range(len(df_enc)),
        test_size=0.10, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_temp, y_temp, temp_idx,
        test_size=0.2222, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    X_train_sc = X_train.copy()
    X_val_sc = X_val.copy()
    X_train_sc[num_cols] = scaler.fit_transform(X_train_sc[num_cols])
    X_val_sc[num_cols] = scaler.transform(X_val_sc[num_cols])

    return X_val_sc, y_val, np.array(val_idx), scaler


# ── Sidebar Navigation ──
st.sidebar.title("CKD Risk Predictor")
st.sidebar.markdown("---")

pages = [
    "Home / Overview",
    "Data Explorer",
    "Model Performance",
    "Risk Assessment",
    "Population Analytics",
    "About",
]
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "*Built for CSE Capstone Project*\n\n"
    "*End-to-End Explainable CKD Risk Prediction*"
)


# ── Helper Functions ──
def risk_badge(category):
    """Return HTML badge for risk category."""
    color = get_risk_color(category)
    return f'<span style="background-color:{color};color:white;padding:5px 15px;border-radius:15px;font-weight:bold;font-size:16px;">{category} Risk</span>'


def stage_badge(stage):
    """Return HTML badge for CKD stage."""
    color = get_stage_color(stage)
    desc = get_stage_description(stage)
    return f'<span style="background-color:{color};color:white;padding:5px 15px;border-radius:15px;font-weight:bold;font-size:14px;">{stage} — {desc}</span>'


# ═══════════════════════════════════════════════════════
# PAGE 1: HOME / OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "Home / Overview":
    st.title("CKD Risk Prediction & Severity Assessment")
    st.markdown("### End-to-End Explainable ML Pipeline for Chronic Kidney Disease")
    st.markdown("---")

    df = load_dataset()
    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", summary["total_patients"])
    col2.metric("CKD Positive", f"{summary['ckd_positive']} ({summary['ckd_percentage']}%)")
    col3.metric("Average Age", summary["avg_age"])
    col4.metric("Avg eGFR", summary["avg_egfr"])

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("CKD Negative", summary["ckd_negative"])
    col6.metric("Avg Creatinine", f"{summary['avg_serum_creatinine']} mg/dL")
    col7.metric("Avg Hemoglobin", f"{summary['avg_hemoglobin']} g/dL")
    col8.metric("Null Values", summary["null_count"])

    st.markdown("---")
    st.subheader("Project Overview")
    st.markdown(
        """
        This project implements a complete machine learning pipeline for:

        - **CKD Prediction**: Binary classification (CKD vs Not CKD)
        - **Risk Scoring**: Probability-based risk categories (Low / Medium / High)
        - **Explainability**: SHAP-based feature explanations for each prediction
        - **Severity Assessment**: eGFR-based CKD staging (Stages 1-5)

        **Models Trained**: LightGBM, CatBoost, XGBoost, Random Forest, Stacking Ensemble

        **Split Strategy**: 70% Train / 20% Validation / 10% Test (Stratified)

        **Clinical Priority**: Optimized for Recall to minimize missed CKD cases.
        """
    )

    # Target distribution
    st.subheader("Target Distribution")
    fig = px.pie(
        values=[summary["ckd_positive"], summary["ckd_negative"]],
        names=["CKD", "Not CKD"],
        color_discrete_sequence=["#e74c3c", "#2ecc71"],
        hole=0.4,
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORER
# ═══════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Data Explorer")
    df = load_dataset()
    df_enc = load_encoded_dataset()

    # Raw data table
    st.subheader("Raw Dataset")
    filter_ckd = st.selectbox("Filter by CKD Diagnosis", ["All", "CKD (1)", "Not CKD (0)"])
    if filter_ckd == "CKD (1)":
        st.dataframe(df[df["ckd_diagnosis"] == 1], use_container_width=True)
    elif filter_ckd == "Not CKD (0)":
        st.dataframe(df[df["ckd_diagnosis"] == 0], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap — drop non-numeric columns
    st.subheader("Correlation Heatmap")
    df_enc_numeric = df_enc.select_dtypes(include=[np.number])
    short = {c: get_display_name(c) for c in df_enc_numeric.columns}
    corr = df_enc_numeric.rename(columns=short).corr()

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature distribution
    st.subheader("Feature Distribution")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    col_choice = st.selectbox(
        "Select Feature", numeric_cols + cat_cols, format_func=get_display_name,
    )

    if col_choice in numeric_cols:
        fig = px.histogram(
            df, x=col_choice, color="ckd_diagnosis", barmode="overlay", nbins=30,
            labels={col_choice: get_display_name(col_choice), "ckd_diagnosis": "CKD"},
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        counts = df.groupby([col_choice, "ckd_diagnosis"]).size().reset_index(name="count")
        fig = px.bar(
            counts, x=col_choice, y="count", color="ckd_diagnosis", barmode="group",
            labels={col_choice: get_display_name(col_choice)},
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance")

    comp_df = load_comparison_table()
    test_df = load_test_comparison_table()

    if comp_df is None:
        st.warning("No model comparison data found. Run the pipeline first (python main.py).")
    else:
        # Validation results
        st.subheader("Validation Set Results (Model Selection)")
        st.dataframe(comp_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="#d4edda",
        ), use_container_width=True)

        # Test results
        if test_df is not None:
            st.subheader("Test Set Results (Unbiased Final Evaluation)")
            st.dataframe(test_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="#d4edda",
            ), use_container_width=True)

        st.markdown("---")

        # Grouped bar chart
        st.subheader("Performance Metrics (Validation)")
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        available = [m for m in metrics if m in comp_df.columns]

        fig = go.Figure()
        for m in available:
            fig.add_trace(go.Bar(name=m, x=comp_df["Model"], y=comp_df[m]))
        fig.update_layout(barmode="group", yaxis_range=[0, 1.05], height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ROC Curves
        st.subheader("ROC Curves (Validation Set)")
        models = load_trained_models()
        if models:
            X_val_sc, y_val, _, _ = get_validation_data()

            fig = go.Figure()
            from sklearn.metrics import roc_curve, roc_auc_score
            for name, model in models.items():
                y_proba = model.predict_proba(X_val_sc.values)[:, 1]
                fpr, tpr, _ = roc_curve(y_val, y_proba)
                auc_val = roc_auc_score(y_val, y_proba)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={auc_val:.3f})"
                ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"), name="Random"
            ))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Confusion Matrix viewer
        st.subheader("Confusion Matrix (Validation Set)")
        if models:
            selected = st.selectbox("Select Model", list(models.keys()))
            model = models[selected]

            X_val_sc, y_val, _, _ = get_validation_data()
            from sklearn.metrics import confusion_matrix
            y_pred = model.predict(X_val_sc.values)
            cm = confusion_matrix(y_val, y_pred)

            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Not CKD", "CKD"], y=["Not CKD", "CKD"],
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Cross-validation
        if "CV Recall Mean" in comp_df.columns:
            st.markdown("---")
            st.subheader("Cross-Validation Results (5-Fold Recall)")
            cv_fig = go.Figure()
            cv_fig.add_trace(go.Bar(
                x=comp_df["Model"], y=comp_df["CV Recall Mean"],
                error_y=dict(type="data", array=comp_df["CV Recall Std"]),
                marker_color="#3498db",
            ))
            cv_fig.update_layout(yaxis_title="Recall", yaxis_range=[0, 1.1], height=400)
            st.plotly_chart(cv_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 4: RISK ASSESSMENT
# ═══════════════════════════════════════════════════════
elif page == "Risk Assessment":
    st.title("Risk Assessment")

    models = load_trained_models()
    comp_df = load_comparison_table()

    if not models or comp_df is None:
        st.warning("Models not trained yet. Run the pipeline first (python main.py).")
    else:
        best_name = comp_df.sort_values(
            by=["Recall", "F1-Score"], ascending=False
        ).iloc[0]["Model"]
        best_model = models[best_name]
        feature_names = load_feature_names()

        st.info(f"Using best model: **{best_name}**")

        mode = st.radio("Assessment Mode", ["Existing Patient", "Manual Input"], horizontal=True)

        if mode == "Existing Patient":
            df = load_dataset()
            X_val_sc, y_val, val_indices, scaler = get_validation_data()

            patient_idx = st.selectbox(
                "Select Patient (Validation Set Index)",
                range(len(X_val_sc)),
                format_func=lambda i: f"Patient {i} (Row {val_indices[i]})",
            )

            if st.button("Assess Risk", type="primary"):
                row = X_val_sc.values[patient_idx].reshape(1, -1)
                pred = int(best_model.predict(row)[0])
                prob = float(best_model.predict_proba(row)[0, 1])
                risk = get_risk_category(prob)

                orig_idx = val_indices[patient_idx]
                orig_row = df.iloc[orig_idx]
                egfr_val = float(orig_row.get("egfr", 0))

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", "CKD" if pred == 1 else "Not CKD")
                c2.metric("Probability", f"{prob*100:.1f}%")
                c3.metric("Actual Label", "CKD" if int(y_val.iloc[patient_idx]) == 1 else "Not CKD")

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Risk Category:** {risk_badge(risk)}", unsafe_allow_html=True)
                with col_b:
                    if pred == 1:
                        stage = classify_egfr_stage(egfr_val)
                        st.markdown(f"**CKD Stage:** {stage_badge(stage)}", unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '**CKD Stage:** <span style="background-color:#95a5a6;color:white;'
                            'padding:5px 15px;border-radius:15px;font-weight:bold;font-size:14px;">'
                            'N/A — No CKD Detected</span>',
                            unsafe_allow_html=True,
                        )

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    title={"text": "CKD Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": get_risk_color(risk)},
                        "steps": [
                            {"range": [0, 30], "color": "#d5f5e3"},
                            {"range": [30, 70], "color": "#fdebd0"},
                            {"range": [70, 100], "color": "#fadbd8"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": prob * 100},
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Demographics & Clinical
                st.markdown("---")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("#### Demographics")
                    st.write(f"**Age:** {orig_row['age']}")
                    st.write(f"**Gender:** {orig_row['gender']}")
                with d2:
                    st.markdown("#### Clinical Values")
                    st.write(f"**Serum Creatinine:** {orig_row['serum_creatinine']} mg/dL")
                    st.write(f"**eGFR:** {egfr_val} mL/min/1.73m²")
                    st.write(f"**Hemoglobin:** {orig_row['hemoglobin']} g/dL")

                # SHAP explanations
                st.markdown("---")
                st.subheader("Prediction Explanation (SHAP)")
                shap_values, X_shap, _ = compute_shap_cache(best_name)
                if shap_values is not None:
                    explanations = explain_patient(shap_values, feature_names, patient_idx, top_n=5)

                    for exp in explanations:
                        color = "#e74c3c" if exp["direction"] == "increases" else "#2ecc71"
                        st.markdown(
                            f'<div style="padding:8px;margin:4px 0;border-left:4px solid {color};'
                            f'background-color:#f8f9fa;border-radius:4px;">'
                            f'<strong>{get_display_name(exp["feature"])}</strong> '
                            f'{exp["direction"]} CKD risk '
                            f'(SHAP = {exp["shap_value"]:+.4f})</div>',
                            unsafe_allow_html=True,
                        )

                    sv = shap_values[patient_idx]
                    sorted_idx = np.argsort(np.abs(sv))[::-1][:10]
                    fig = go.Figure(go.Bar(
                        y=[get_display_name(feature_names[i]) for i in sorted_idx][::-1],
                        x=[sv[i] for i in sorted_idx][::-1],
                        orientation="h",
                        marker_color=["#e74c3c" if sv[i] > 0 else "#2ecc71" for i in sorted_idx][::-1],
                    ))
                    fig.update_layout(
                        title="SHAP Values for This Patient",
                        xaxis_title="SHAP Value (impact on prediction)",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:  # Manual input
            st.subheader("Enter Patient Data")
            st.caption(
                "Serum creatinine is used to automatically calculate eGFR "
                "(CKD-EPI 2021 formula). eGFR is used for CKD staging only — "
                "the model predicts CKD from symptoms, demographics, and hemoglobin."
            )

            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=45)
                gender = st.selectbox("Gender", ["Male", "Female"])
                hemoglobin = st.number_input(
                    "Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.5, step=0.1
                )
                serum_creatinine_input = st.number_input(
                    "Serum Creatinine (mg/dL)",
                    min_value=0.1, max_value=15.0, value=1.0, step=0.1,
                    help="Blood test result. eGFR will be auto-calculated from Age, Gender, and this value.",
                )

                # Auto-calculate eGFR using CKD-EPI 2021 formula
                from src.staging import estimate_egfr_ckd_epi
                egfr_input = estimate_egfr_ckd_epi(
                    serum_creatinine=serum_creatinine_input,
                    age=float(age),
                    is_female=(gender == "Female"),
                )
                st.info(
                    f"**Auto-calculated eGFR:** {egfr_input} mL/min/1.73m² "
                    f"(CKD-EPI 2021 formula)"
                )

            with col2:
                diabetes = st.selectbox("Diagnosed with Diabetes?", ["No", "Yes"])
                hypertension = st.selectbox("High Blood Pressure?", ["No", "Yes"])
                foamy_urine = st.selectbox("Foamy Urine?", ["No", "Yes"])
                extra_salt = st.selectbox("Extra Salt Intake?", ["No", "Yes"])

            if st.button("Predict CKD Risk", type="primary"):
                yes_no = lambda v: 1 if v == "Yes" else 0

                input_data = {
                    "age": age,
                    "gender": 1 if gender == "Male" else 0,
                    "have_you_ever_been_diagnosed_with_diabetes": yes_no(diabetes),
                    "do_you_have_high_blood_pressure_hypertension": yes_no(hypertension),
                    "do_you_notice_foamy_urine": yes_no(foamy_urine),
                    "do_you_take_extra_salt_with_your_food": yes_no(extra_salt),
                    "hemoglobin": hemoglobin,
                }

                # Scale numerical features using training scaler
                _, _, _, scaler = get_validation_data()

                # Create ordered vector matching training feature order
                input_vector = np.array([input_data[f] for f in feature_names], dtype=float).reshape(1, -1)

                # Scale numerical columns
                num_cols = [c for c in NUMERICAL_FEATURES if c in feature_names]
                num_indices = [feature_names.index(c) for c in num_cols]
                num_vals = input_vector[0, num_indices].reshape(1, -1)
                scaled_vals = scaler.transform(num_vals)
                for i, idx in enumerate(num_indices):
                    input_vector[0, idx] = scaled_vals[0, i]

                # Predict
                pred = int(best_model.predict(input_vector)[0])
                prob = float(best_model.predict_proba(input_vector)[0, 1])
                risk = get_risk_category(prob)

                st.markdown("---")
                st.subheader("Results")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction", "CKD" if pred == 1 else "Not CKD")
                c2.metric("Probability", f"{prob*100:.1f}%")
                c3.metric("eGFR", f"{egfr_input}", help="Auto-calculated from CKD-EPI 2021 formula")
                if pred == 1:
                    stage = classify_egfr_stage(egfr_input)
                    c4.metric("CKD Stage", f"{stage}")
                else:
                    c4.metric("CKD Stage", "N/A")

                st.markdown(f"**Risk Category:** {risk_badge(risk)}", unsafe_allow_html=True)
                st.markdown(
                    f"**Serum Creatinine:** {serum_creatinine_input} mg/dL &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Computed eGFR:** {egfr_input} mL/min/1.73m²"
                )
                if pred == 1:
                    stage = classify_egfr_stage(egfr_input)
                    st.markdown(f"**CKD Stage:** {stage_badge(stage)}", unsafe_allow_html=True)
                else:
                    st.markdown(
                        '**CKD Stage:** <span style="background-color:#95a5a6;color:white;'
                        'padding:5px 15px;border-radius:15px;font-weight:bold;font-size:14px;">'
                        'N/A — No CKD Detected</span>',
                        unsafe_allow_html=True,
                    )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    title={"text": "CKD Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": get_risk_color(risk)},
                        "steps": [
                            {"range": [0, 30], "color": "#d5f5e3"},
                            {"range": [30, 70], "color": "#fdebd0"},
                            {"range": [70, 100], "color": "#fadbd8"},
                        ],
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 5: POPULATION ANALYTICS
# ═══════════════════════════════════════════════════════
elif page == "Population Analytics":
    st.title("Population Analytics")
    df = load_dataset()
    risk_df = load_risk_scores()
    staging_df = load_staging_data()

    if risk_df is None or staging_df is None:
        st.warning("Run the pipeline first to generate analytics data (python main.py).")
    else:
        st.subheader("Risk Distribution (Validation Set)")
        c1, c2 = st.columns(2)
        with c1:
            counts = risk_df["Risk_Category"].value_counts()
            fig = px.pie(
                values=counts.values, names=counts.index, color=counts.index,
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                hole=0.4,
            )
            fig.update_layout(height=350, title="Risk Categories")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                risk_df, x="CKD_Probability", nbins=25,
                color_discrete_sequence=["#3498db"],
                labels={"CKD_Probability": "CKD Probability"},
            )
            fig.add_vline(x=0.3, line_dash="dash", line_color="#f39c12", annotation_text="Low/Med")
            fig.add_vline(x=0.7, line_dash="dash", line_color="#e74c3c", annotation_text="Med/High")
            fig.update_layout(height=350, title="Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("CKD Stage Distribution (Full Dataset)")
        stage_order = ["Stage 1", "Stage 2", "Stage 3a", "Stage 3b", "Stage 4", "Stage 5"]
        stage_counts = staging_df["CKD_Stage"].value_counts().reindex(stage_order, fill_value=0)
        fig = px.bar(
            x=stage_counts.index, y=stage_counts.values,
            color=stage_counts.index,
            color_discrete_map={s: get_stage_color(s) for s in stage_order},
            labels={"x": "CKD Stage", "y": "Count"},
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Feature Importance (SHAP)")
        comp_df = load_comparison_table()
        if comp_df is not None:
            best_name = comp_df.sort_values(
                by=["Recall", "F1-Score"], ascending=False
            ).iloc[0]["Model"]
            feature_names = load_feature_names()
            shap_values, _, _ = compute_shap_cache(best_name)

            if shap_values is not None and feature_names is not None:
                mean_shap = np.abs(shap_values).mean(axis=0)
                sorted_idx = np.argsort(mean_shap)[::-1]

                fig = go.Figure(go.Bar(
                    y=[get_display_name(feature_names[i]) for i in sorted_idx][::-1],
                    x=[mean_shap[i] for i in sorted_idx][::-1],
                    orientation="h", marker_color="#e74c3c",
                ))
                fig.update_layout(
                    title="Mean |SHAP| Feature Importance",
                    xaxis_title="Mean |SHAP Value|", height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Demographic Breakdowns")
        tab1, tab2 = st.tabs(["By Gender", "By Age Group"])

        with tab1:
            gender_ckd = df.groupby(["gender", "ckd_diagnosis"]).size().reset_index(name="count")
            fig = px.bar(
                gender_ckd, x="gender", y="count", color="ckd_diagnosis", barmode="group",
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                labels={"gender": "Gender", "ckd_diagnosis": "CKD"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            df_temp = df.copy()
            df_temp["Age_Group"] = pd.cut(
                df_temp["age"], bins=[0, 30, 40, 50, 60, 70, 120],
                labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"],
            )
            age_ckd = df_temp.groupby(["Age_Group", "ckd_diagnosis"]).size().reset_index(name="count")
            fig = px.bar(
                age_ckd, x="Age_Group", y="count", color="ckd_diagnosis", barmode="group",
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                labels={"Age_Group": "Age Group", "ckd_diagnosis": "CKD"},
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 6: ABOUT
# ═══════════════════════════════════════════════════════
elif page == "About":
    st.title("About This Project")

    st.markdown(
        """
        ### End-to-End Explainable CKD Risk Prediction & Severity Assessment

        **Course:** CSE Capstone Project

        ---

        ### Methodology

        1. **Data Collection**: 2159 patients, locally collected clinical data
        2. **Preprocessing**: Encoding, scaling, SMOTE for class imbalance
        3. **Stratified Split**: 70% Training / 20% Validation / 10% Test
        4. **Model Training**: LightGBM, CatBoost, XGBoost, Random Forest, Stacking Ensemble
        5. **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cross-Validation
        6. **Risk Scoring**: Probability-based categorization (Low/Medium/High)
        7. **Explainability**: SHAP (SHapley Additive exPlanations)
        8. **Severity Assessment**: eGFR-based CKD staging (Stages 1-5)

        ### Key Design Decisions

        - **Clinical Priority**: Models optimized for Recall (minimizing false negatives)
        - **eGFR & Serum Creatinine Excluded**: eGFR defines CKD clinically; serum creatinine is used to compute eGFR. Including either would be data leakage. Both are used only for CKD staging.
        - **Class Imbalance**: Handled via SMOTE on training data only
        - **No Data Leakage**: Scaler and SMOTE fitted on training set only; validation/test sets never seen during training
        - **Reproducibility**: Random seed = 42 throughout

        ---

        **License:** MIT

        ---

        *This tool is for research and decision support purposes only.
        It is not intended to replace professional medical diagnosis or treatment.*
        """
    )
