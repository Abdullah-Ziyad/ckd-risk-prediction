"""
CKD Risk Prediction Pipeline — CLI Runner
Executes all modules sequentially for end-to-end ML pipeline.

Dataset: final_verified_ckd_dataset_fixed.xlsx (2159 patients)
Stratified Split: 70% Train / 20% Validation / 10% Test
eGFR excluded from prediction features (used only for staging).
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

# Suppress common warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set global random seed
np.random.seed(42)


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CKD Risk Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/final_verified_ckd_dataset_fixed.xlsx",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--patient_index",
        type=int,
        default=0,
        help="Patient index for sample report (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete CKD risk prediction pipeline."""
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args()

    print("\n" + "=" * 60)
    print("  CKD RISK PREDICTION & SEVERITY ASSESSMENT PIPELINE")
    print("=" * 60)

    # ── Module 1: Data Loading ──
    print("\n" + "─" * 50)
    print("  MODULE 1: Loading Data")
    print("─" * 50)
    from src.data_loader import load_data, get_dataset_summary

    df = load_data(args.data_path)
    summary = get_dataset_summary(df)
    print(f"  Total patients: {summary['total_patients']}")
    print(f"  CKD positive:   {summary['ckd_positive']} ({summary['ckd_percentage']}%)")
    print(f"  CKD negative:   {summary['ckd_negative']}")
    print(f"  Null values:    {summary['null_count']}")

    # Keep a copy of original data before encoding
    df_original = df.copy()

    # ── Module 2: Preprocessing ──
    print("\n" + "─" * 50)
    print("  MODULE 2: Preprocessing (70/20/10 Stratified Split)")
    print("─" * 50)
    from src.preprocessing import preprocess, encode_features, EXCLUDE_FROM_PREDICTION

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = preprocess(df)
    print(f"  Training samples (after SMOTE): {len(X_train)}")
    print(f"  Validation samples:             {len(X_val)}")
    print(f"  Test samples:                   {len(X_test)}")
    print(f"  Prediction features:            {len(feature_names)}")
    print(f"  Excluded from prediction:       {EXCLUDE_FROM_PREDICTION}")

    # Get validation indices for report generation
    from sklearn.model_selection import train_test_split
    df_encoded = encode_features(df_original.copy())
    all_indices = np.arange(len(df_encoded))

    _, test_split_idx, _, _ = train_test_split(
        all_indices, df_encoded["ckd_diagnosis"],
        test_size=0.10, random_state=42, stratify=df_encoded["ckd_diagnosis"],
    )
    remaining_idx = np.setdiff1d(all_indices, test_split_idx)
    remaining_y = df_encoded["ckd_diagnosis"].iloc[remaining_idx]

    _, val_split_idx, _, _ = train_test_split(
        remaining_idx, remaining_y,
        test_size=0.2222, random_state=42, stratify=remaining_y,
    )

    val_indices = np.array(val_split_idx)

    # ── Module 3: Model Training ──
    print("\n" + "─" * 50)
    print("  MODULE 3: Training Models")
    print("─" * 50)
    from src.model_training import train_all_models

    models = train_all_models(X_train, y_train, feature_names)
    print(f"  Trained {len(models)} models: {list(models.keys())}")

    # ── Module 4: Evaluation ──
    print("\n" + "─" * 50)
    print("  MODULE 4: Evaluating Models")
    print("─" * 50)
    from src.evaluation import evaluate_models

    best_name, best_model, val_comparison_df, test_results_df = evaluate_models(
        models, X_val, y_val, X_test, y_test, X_train, y_train
    )
    print(f"\n  Best Model (selected on validation): {best_name}")
    print(f"\n  Validation Set Results:")
    print(val_comparison_df.to_string(index=False))
    print(f"\n  Test Set Results (unbiased):")
    print(test_results_df.to_string(index=False))

    # ── Module 5: Risk Scoring ──
    print("\n" + "─" * 50)
    print("  MODULE 5: Risk Scoring")
    print("─" * 50)
    from src.risk_scoring import assign_risk_scores

    risk_df = assign_risk_scores(best_model, X_val, y_val)
    print(f"  Risk distribution (validation set):")
    print(f"    {risk_df['Risk_Category'].value_counts().to_dict()}")

    # ── Module 6: SHAP Explainability ──
    print("\n" + "─" * 50)
    print("  MODULE 6: SHAP Explainability")
    print("─" * 50)
    from src.explainability import compute_shap_values, explain_patient

    # SHAP TreeExplainer doesn't support StackingClassifier — use best base model
    shap_model_name = best_name
    shap_model = best_model
    if best_name == "Stacking":
        # Use the best non-stacking model for SHAP
        base_df = val_comparison_df[val_comparison_df["Model"] != "Stacking"]
        base_sorted = base_df.sort_values(by=["Recall", "F1-Score"], ascending=False)
        shap_model_name = base_sorted.iloc[0]["Model"]
        shap_model = models[shap_model_name]
        print(f"  (Using {shap_model_name} for SHAP — TreeExplainer does not support Stacking)")

    shap_values, explainer = compute_shap_values(
        shap_model, X_val, feature_names
    )
    sample_explanations = explain_patient(
        shap_values, feature_names, args.patient_index
    )
    print(f"  Top explanations for patient {args.patient_index}:")
    for exp in sample_explanations:
        print(f"    - {exp['description']}")

    # ── Module 7: CKD Staging ──
    print("\n" + "─" * 50)
    print("  MODULE 7: CKD Staging")
    print("─" * 50)
    from src.staging import compute_staging

    staging_df = compute_staging(df_original)
    print(f"  Stage distribution:")
    print(f"    {staging_df['CKD_Stage'].value_counts().to_dict()}")

    # ── Module 8: Reporting ──
    print("\n" + "─" * 50)
    print("  MODULE 8: Generating Reports & Visualizations")
    print("─" * 50)
    from src.reporting import (
        generate_patient_report,
        plot_correlation_heatmap,
        plot_feature_importance,
        plot_dashboard_summary,
    )

    # Correlation heatmap (exclude egfr and ckd_stage)
    df_enc_for_corr = df_encoded.drop(columns=EXCLUDE_FROM_PREDICTION, errors="ignore")
    plot_correlation_heatmap(df_enc_for_corr)

    # Feature importance (use base model that has feature_importances_)
    fi_model = shap_model if not hasattr(best_model, "feature_importances_") else best_model
    plot_feature_importance(fi_model, feature_names)

    # Dashboard summary
    plot_dashboard_summary(val_comparison_df, risk_df, staging_df)

    # Patient report
    report = generate_patient_report(
        patient_idx=args.patient_index,
        df_original=df_original,
        X_test=X_val,
        y_test=y_val,
        model=best_model,
        shap_values=shap_values,
        feature_names=feature_names,
        staging_df=staging_df,
        test_indices=val_indices,
    )

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Dataset:         {args.data_path}")
    print(f"  Total patients:  {summary['total_patients']}")
    print(f"  Split:           70% Train / 20% Validation / 10% Test")
    print(f"  Best Model:      {best_name}")
    best_val = val_comparison_df[val_comparison_df["Model"] == best_name].iloc[0]
    best_test = test_results_df[test_results_df["Model"] == best_name].iloc[0]
    print(f"\n  Validation Metrics:")
    print(f"    Accuracy:      {best_val['Accuracy']:.4f}")
    print(f"    Recall:        {best_val['Recall']:.4f}")
    print(f"    F1-Score:      {best_val['F1-Score']:.4f}")
    print(f"    ROC-AUC:       {best_val['ROC-AUC']:.4f}")
    print(f"\n  Test Metrics (unbiased):")
    print(f"    Accuracy:      {best_test['Accuracy']:.4f}")
    print(f"    Recall:        {best_test['Recall']:.4f}")
    print(f"    F1-Score:      {best_test['F1-Score']:.4f}")
    print(f"    ROC-AUC:       {best_test['ROC-AUC']:.4f}")
    print()
    print(f"  Sample Patient Report (index={args.patient_index}):")
    print(f"    Prediction:    {report['prediction']['ckd_diagnosis']}")
    print(f"    Probability:   {report['prediction']['probability']}%")
    print(f"    Risk:          {report['prediction']['risk_category']}")
    print(f"    CKD Stage:     {report['staging']['ckd_stage']} ({report['staging']['stage_description']})")
    print(f"    Actual Label:  {report['actual_label']}")
    print()
    print("  Outputs saved to: outputs/")
    print("  Models saved to:  models/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
