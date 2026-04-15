"""
Module 6: SHAP Explainability
Generates global and local SHAP explanations for model predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str = "outputs",
) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """Compute SHAP values and generate explanation plots.

    Args:
        model: Trained tree-based model.
        X_test: Test feature matrix.
        feature_names: List of feature names.
        output_dir: Directory to save plots.

    Returns:
        shap_values (array), explainer (TreeExplainer)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classifiers that return a list, take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info("SHAP values shape: %s", shap_values.shape)

    # Generate global plots
    _plot_summary_beeswarm(shap_values, X_test, feature_names, out)
    _plot_bar(shap_values, feature_names, out)

    # Generate sample patient plots (index 0)
    _plot_waterfall(explainer, shap_values, X_test, feature_names, 0, out)
    _plot_force(explainer, shap_values, X_test, feature_names, 0, out)

    return shap_values, explainer


def explain_patient(
    shap_values: np.ndarray,
    feature_names: List[str],
    patient_index: int,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Get top contributing features for a specific patient.

    Args:
        shap_values: SHAP values array.
        feature_names: List of feature names.
        patient_index: Index of the patient in the test set.
        top_n: Number of top features to return.

    Returns:
        List of dicts with feature name, SHAP value, and direction.
    """
    patient_shap = shap_values[patient_index]
    abs_shap = np.abs(patient_shap)
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    explanations = []
    for idx in top_indices:
        value = float(patient_shap[idx])
        explanations.append(
            {
                "feature": feature_names[idx],
                "shap_value": round(value, 4),
                "direction": "increases" if value > 0 else "decreases",
                "description": (
                    f"{feature_names[idx]} {'increases' if value > 0 else 'decreases'} "
                    f"CKD risk (SHAP = {value:+.4f})"
                ),
            }
        )

    return explanations


def _plot_summary_beeswarm(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    out: Path,
) -> None:
    """Generate SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, show=False, plot_size=None
    )
    plt.title("SHAP Feature Importance (Beeswarm)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP beeswarm summary plot.")


def _plot_bar(
    shap_values: np.ndarray, feature_names: List[str], out: Path
) -> None:
    """Generate SHAP bar plot (mean |SHAP| values)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(len(sorted_idx)),
        mean_abs[sorted_idx],
        color="#e74c3c",
        edgecolor="black",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (Mean |SHAP|)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "shap_bar_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP bar plot.")


def _plot_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    patient_idx: int,
    out: Path,
) -> None:
    """Generate SHAP waterfall plot for a single patient."""
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1] if len(expected) > 1 else expected[0]

        explanation = shap.Explanation(
            values=shap_values[patient_idx],
            base_values=expected,
            data=X_test[patient_idx],
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"SHAP Waterfall — Patient {patient_idx}", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            out / f"shap_waterfall_patient_{patient_idx}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")
        logger.info("Saved SHAP waterfall plot for patient %d.", patient_idx)
    except Exception as e:
        logger.warning("Could not generate waterfall plot: %s", e)


def _plot_force(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    patient_idx: int,
    out: Path,
) -> None:
    """Generate SHAP force plot and save as HTML."""
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1] if len(expected) > 1 else expected[0]

        force_plot = shap.force_plot(
            expected,
            shap_values[patient_idx],
            X_test[patient_idx],
            feature_names=feature_names,
            matplotlib=False,
        )
        shap.save_html(
            str(out / f"shap_force_patient_{patient_idx}.html"), force_plot
        )
        logger.info("Saved SHAP force plot (HTML) for patient %d.", patient_idx)
    except Exception as e:
        logger.warning("Could not generate force plot: %s", e)
