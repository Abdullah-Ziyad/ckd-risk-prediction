"""
Module 8: Reporting & Visualization
Generates patient report cards, correlation heatmaps, feature importance plots,
and a combined dashboard summary.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_display_name
from src.risk_scoring import get_risk_category, get_risk_color
from src.staging import classify_egfr_stage, get_stage_description
from src.explainability import explain_patient

logger = logging.getLogger(__name__)


def generate_patient_report(
    patient_idx: int,
    df_original: pd.DataFrame,
    X_test: np.ndarray,
    y_test: pd.Series,
    model: Any,
    shap_values: np.ndarray,
    feature_names: List[str],
    staging_df: pd.DataFrame,
    test_indices: np.ndarray,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Generate a comprehensive patient report card.

    Args:
        patient_idx: Index within the test set.
        df_original: Original (unencoded) DataFrame.
        X_test: Test feature matrix.
        y_test: Test labels.
        model: Trained best model.
        shap_values: SHAP values array.
        feature_names: Feature name list.
        staging_df: DataFrame with eGFR staging.
        test_indices: Original DataFrame indices of test set rows.
        output_dir: Directory to save report.

    Returns:
        Dictionary containing the full patient report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Get original row
    original_idx = test_indices[patient_idx]
    patient_row = df_original.iloc[original_idx]

    # Prediction
    prediction = int(model.predict(X_test[patient_idx].reshape(1, -1))[0])
    probability = float(model.predict_proba(X_test[patient_idx].reshape(1, -1))[0, 1])
    risk_category = get_risk_category(probability)

    # CKD Stage — only show for CKD-positive predictions
    egfr_value = float(patient_row.get("egfr", 0))
    if prediction == 1:
        ckd_stage = classify_egfr_stage(egfr_value)
        stage_desc = get_stage_description(ckd_stage)
    else:
        ckd_stage = "N/A"
        stage_desc = "No CKD detected"

    # SHAP explanations
    top_explanations = explain_patient(shap_values, feature_names, patient_idx, top_n=3)

    report = {
        "patient_index": patient_idx,
        "original_index": int(original_idx),
        "demographics": {
            "age": int(patient_row.get("age", 0)),
            "gender": str(patient_row.get("gender", "Unknown")),
        },
        "clinical_values": {
            "serum_creatinine": float(patient_row.get("serum_creatinine", 0)),
            "egfr": egfr_value,
            "hemoglobin": float(patient_row.get("hemoglobin", 0)),
        },
        "prediction": {
            "ckd_diagnosis": "CKD" if prediction == 1 else "Not CKD",
            "probability": round(probability * 100, 2),
            "risk_category": risk_category,
        },
        "staging": {
            "egfr": egfr_value,
            "ckd_stage": ckd_stage,
            "stage_description": stage_desc,
        },
        "explanations": top_explanations,
        "actual_label": "CKD" if int(y_test.iloc[patient_idx]) == 1 else "Not CKD",
    }

    # Format and save report
    report_text = _format_report(report)
    report_path = out / f"patient_report_{patient_idx}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info("Patient report saved to %s", report_path)

    # Also save as CSV row
    flat = {
        "Patient_Index": patient_idx,
        "Age": report["demographics"]["age"],
        "Gender": report["demographics"]["gender"],
        "Serum_Creatinine": report["clinical_values"]["serum_creatinine"],
        "eGFR": report["clinical_values"]["egfr"],
        "Hemoglobin": report["clinical_values"]["hemoglobin"],
        "Prediction": report["prediction"]["ckd_diagnosis"],
        "CKD_Probability_%": report["prediction"]["probability"],
        "Risk_Category": report["prediction"]["risk_category"],
        "CKD_Stage": report["staging"]["ckd_stage"],
        "Stage_Description": report["staging"]["stage_description"],
        "Actual_Label": report["actual_label"],
    }
    for i, exp in enumerate(top_explanations, 1):
        flat[f"Top{i}_Feature"] = exp["feature"]
        flat[f"Top{i}_SHAP"] = exp["shap_value"]
        flat[f"Top{i}_Direction"] = exp["direction"]

    pd.DataFrame([flat]).to_csv(out / f"patient_report_{patient_idx}.csv", index=False)

    return report


def _format_report(report: Dict[str, Any]) -> str:
    """Format a patient report as readable text."""
    lines = [
        "=" * 60,
        "       CKD RISK PREDICTION — PATIENT REPORT CARD",
        "=" * 60,
        "",
        f"  Patient Index (Test Set): {report['patient_index']}",
        f"  Original Dataset Row:     {report['original_index']}",
        "",
        "--- DEMOGRAPHICS ---",
        f"  Age:         {report['demographics']['age']}",
        f"  Gender:      {report['demographics']['gender']}",
        "",
        "--- CLINICAL VALUES ---",
        f"  Serum Creatinine: {report['clinical_values']['serum_creatinine']} mg/dL",
        f"  eGFR:             {report['clinical_values']['egfr']} mL/min/1.73m²",
        f"  Hemoglobin:       {report['clinical_values']['hemoglobin']} g/dL",
        "",
        "--- ML PREDICTION ---",
        f"  Prediction:   {report['prediction']['ckd_diagnosis']}",
        f"  CKD Probability: {report['prediction']['probability']}%",
        f"  Risk Category:   {report['prediction']['risk_category']}",
        "",
        "--- CKD STAGING ---",
        f"  eGFR:        {report['staging']['egfr']} mL/min/1.73m²",
        f"  CKD Stage:   {report['staging']['ckd_stage']}",
        f"  Description: {report['staging']['stage_description']}",
        "",
        "--- TOP EXPLANATIONS (SHAP) ---",
    ]
    for i, exp in enumerate(report["explanations"], 1):
        lines.append(f"  {i}. {exp['description']}")

    lines.extend(
        [
            "",
            f"  Actual Label: {report['actual_label']}",
            "",
            "=" * 60,
            "  DISCLAIMER: This tool is for decision support only.",
            "  Not a replacement for medical diagnosis.",
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def plot_correlation_heatmap(
    df_encoded: pd.DataFrame, output_dir: str = "outputs"
) -> None:
    """Generate and save correlation heatmap.

    Args:
        df_encoded: Numerically encoded DataFrame.
        output_dir: Directory to save plot.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use short display names
    short_names = {c: get_display_name(c) for c in df_encoded.columns}
    df_plot = df_encoded.rename(columns=short_names)

    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df_plot.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap.")


def plot_feature_importance(
    model: Any, feature_names: List[str], output_dir: str = "outputs"
) -> None:
    """Plot feature importance from the best model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        output_dir: Directory to save plot.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ — skipping plot.")
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    display = [get_display_name(feature_names[i]) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(display, importances[sorted_idx], color="#2980b9", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Best Model)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot.")


def plot_dashboard_summary(
    comparison_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    staging_df: pd.DataFrame,
    output_dir: str = "outputs",
) -> None:
    """Generate a combined dashboard summary plot.

    Args:
        comparison_df: Model comparison table.
        risk_df: Risk scoring DataFrame.
        staging_df: CKD staging DataFrame.
        output_dir: Directory to save plot.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Model comparison
    ax = axes[0, 0]
    metrics = ["Accuracy", "Recall", "F1-Score", "ROC-AUC"]
    available = [m for m in metrics if m in comparison_df.columns]
    x = np.arange(len(comparison_df))
    width = 0.2
    for i, m in enumerate(available):
        ax.bar(x + i * width, comparison_df[m], width, label=m)
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right", fontsize=9)
    ax.set_title("Model Performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # 2. Risk distribution
    ax = axes[0, 1]
    risk_counts = risk_df["Risk_Category"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    colors = [get_risk_color(c) for c in risk_counts.index]
    ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%", colors=colors)
    ax.set_title("Risk Distribution")

    # 3. CKD stage distribution
    ax = axes[1, 0]
    stage_order = ["Stage 1", "Stage 2", "Stage 3a", "Stage 3b", "Stage 4", "Stage 5"]
    stage_counts = staging_df["CKD_Stage"].value_counts().reindex(
        stage_order, fill_value=0
    )
    from src.staging import get_stage_color
    stage_colors = [get_stage_color(s) for s in stage_counts.index]
    ax.bar(stage_counts.index, stage_counts.values, color=stage_colors, edgecolor="black")
    ax.set_title("CKD Stage Distribution")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    # 4. Probability histogram
    ax = axes[1, 1]
    ax.hist(risk_df["CKD_Probability"], bins=20, color="#3498db", edgecolor="black", alpha=0.8)
    ax.axvline(0.3, color="#f39c12", linestyle="--", linewidth=2)
    ax.axvline(0.7, color="#e74c3c", linestyle="--", linewidth=2)
    ax.set_title("CKD Probability Distribution")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    plt.suptitle("CKD Risk Prediction — Dashboard Summary", fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(out / "dashboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dashboard summary plot.")
