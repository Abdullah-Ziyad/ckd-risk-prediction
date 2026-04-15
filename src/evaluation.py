"""
Module 4: Evaluation
Evaluates all models on the validation set for comparison and selection,
then reports final unbiased metrics on the held-out test set.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("outputs")
RANDOM_STATE = 42


def evaluate_models(
    models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    output_dir: str = "outputs",
) -> Tuple[str, Any, pd.DataFrame, pd.DataFrame]:
    """Evaluate all models and select the best based on Recall then F1.

    Uses the VALIDATION set for model comparison and selection.
    Reports final unbiased metrics on the TEST set for all models.

    Args:
        models: Dictionary of trained models.
        X_val: Validation feature matrix (for model comparison).
        y_val: Validation labels.
        X_test: Test feature matrix (for final unbiased evaluation).
        y_test: Test labels.
        X_train: Training features (for cross-validation).
        y_train: Training labels (for cross-validation).
        output_dir: Directory to save outputs.

    Returns:
        best_model_name, best_model, val_comparison_df, test_results_df
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Evaluate all models on VALIDATION set (for comparison) ──
    val_results = []
    for name, model in models.items():
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_proba)

        val_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        })
        logger.info(
            "[VAL] %s — Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%.4f",
            name, acc, prec, rec, f1, auc,
        )

    val_comparison_df = pd.DataFrame(val_results)

    # ── 5-Fold Stratified Cross-Validation on training data ──
    cv_results = []
    if X_train is not None and y_train is not None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for name, model in models.items():
            scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="recall", n_jobs=-1
            )
            cv_results.append({
                "Model": name,
                "CV Recall Mean": round(scores.mean(), 4),
                "CV Recall Std": round(scores.std(), 4),
            })
            logger.info(
                "%s — CV Recall: %.4f ± %.4f", name, scores.mean(), scores.std()
            )

        cv_df = pd.DataFrame(cv_results)
        val_comparison_df = val_comparison_df.merge(cv_df, on="Model", how="left")

    # ── Select best model: prioritize Recall, then F1 ──
    val_sorted = val_comparison_df.sort_values(
        by=["Recall", "F1-Score"], ascending=False
    )
    best_model_name = val_sorted.iloc[0]["Model"]
    best_model = models[best_model_name]
    logger.info("Best model (selected on validation set): %s", best_model_name)

    # ── Evaluate all models on TEST set (final unbiased metrics) ──
    test_results = []
    for name, model in models.items():
        y_pred_t = model.predict(X_test)
        y_proba_t = model.predict_proba(X_test)[:, 1]

        test_results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred_t), 4),
            "Precision": round(precision_score(y_test, y_pred_t, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_t, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred_t, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba_t), 4),
        })
        logger.info(
            "[TEST] %s — Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, AUC=%.4f",
            name,
            test_results[-1]["Accuracy"],
            test_results[-1]["Precision"],
            test_results[-1]["Recall"],
            test_results[-1]["F1-Score"],
            test_results[-1]["ROC-AUC"],
        )

    test_results_df = pd.DataFrame(test_results)

    # ── Save comparison tables ──
    val_comparison_df.to_csv(out / "model_comparison_validation.csv", index=False)
    test_results_df.to_csv(out / "model_comparison_test.csv", index=False)
    # Also save a combined "model_comparison.csv" for dashboard compatibility
    val_comparison_df.to_csv(out / "model_comparison.csv", index=False)
    logger.info("Comparison tables saved to %s", out)

    # ── Generate plots ──
    _plot_comparison_bar(val_comparison_df, out, "Validation")
    _plot_comparison_bar(test_results_df, out, "Test")
    _plot_roc_curves(models, X_val, y_val, out, suffix="validation")
    _plot_roc_curves(models, X_test, y_test, out, suffix="test")
    _plot_confusion_matrices(models, X_val, y_val, out, suffix="validation")
    _plot_confusion_matrices(models, X_test, y_test, out, suffix="test")

    return best_model_name, best_model, val_comparison_df, test_results_df


def _plot_comparison_bar(df: pd.DataFrame, out: Path, set_name: str = "Validation") -> None:
    """Generate grouped bar chart comparing model metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    available = [m for m in metrics if m in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.15

    for i, metric in enumerate(available):
        ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Performance Comparison ({set_name} Set)")
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = f"model_comparison_bar_{set_name.lower()}.png"
    fig.savefig(out / fname, dpi=150)
    # Also save as default name for dashboard
    if set_name == "Validation":
        fig.savefig(out / "model_comparison_bar.png", dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison bar chart (%s).", set_name)


def _plot_roc_curves(
    models: Dict[str, Any], X: np.ndarray, y: np.ndarray, out: Path,
    suffix: str = "validation",
) -> None:
    """Plot ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_val = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    title_set = "Validation" if suffix == "validation" else "Test"
    ax.set_title(f"ROC Curves — All Models ({title_set} Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / f"roc_curves_{suffix}.png", dpi=150)
    # Also save default name
    if suffix == "validation":
        fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves plot (%s).", suffix)


def _plot_confusion_matrices(
    models: Dict[str, Any], X: np.ndarray, y: np.ndarray, out: Path,
    suffix: str = "validation",
) -> None:
    """Plot confusion matrix heatmaps for each model."""
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=["Not CKD", "CKD"],
            yticklabels=["Not CKD", "CKD"],
        )
        axes[idx].set_title(f"{name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    title_set = "Validation" if suffix == "validation" else "Test"
    plt.suptitle(f"Confusion Matrices ({title_set} Set)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out / f"confusion_matrices_{suffix}.png", dpi=150, bbox_inches="tight")
    if suffix == "validation":
        fig.savefig(out / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix heatmaps (%s).", suffix)
