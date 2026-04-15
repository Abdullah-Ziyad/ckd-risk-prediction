"""
Module 5: Risk Scoring
Assigns CKD risk categories based on model prediction probabilities.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def assign_risk_scores(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Compute CKD probabilities and assign risk categories.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test feature matrix.
        y_test: Test labels.
        output_dir: Directory to save plots and CSV.

    Returns:
        DataFrame with patient index, probability, risk category, and actual label.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    probabilities = model.predict_proba(X_test)[:, 1]

    risk_df = pd.DataFrame(
        {
            "Patient_Index": range(len(probabilities)),
            "CKD_Probability": np.round(probabilities, 4),
            "Risk_Category": pd.cut(
                probabilities,
                bins=[-0.01, 0.3, 0.7, 1.01],
                labels=["Low", "Medium", "High"],
            ),
            "Actual_Label": y_test.values if hasattr(y_test, "values") else y_test,
        }
    )

    logger.info("Risk distribution:\n%s", risk_df["Risk_Category"].value_counts().to_string())

    # Save risk scores
    risk_df.to_csv(out / "risk_scores.csv", index=False)
    logger.info("Risk scores saved to %s", out / "risk_scores.csv")

    # Generate plots
    _plot_risk_pie(risk_df, out)
    _plot_risk_bar(risk_df, out)
    _plot_risk_histogram(risk_df, out)

    return risk_df


def get_risk_category(probability: float) -> str:
    """Get risk category for a single probability value."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"


def get_risk_color(category: str) -> str:
    """Return color associated with risk category."""
    colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    return colors.get(category, "#95a5a6")


def _plot_risk_pie(risk_df: pd.DataFrame, out: Path) -> None:
    """Generate risk distribution pie chart."""
    counts = risk_df["Risk_Category"].value_counts()
    colors = [get_risk_color(cat) for cat in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title("CKD Risk Distribution", fontsize=14)
    plt.tight_layout()
    fig.savefig(out / "risk_distribution_pie.png", dpi=150)
    plt.close(fig)
    logger.info("Saved risk distribution pie chart.")


def _plot_risk_bar(risk_df: pd.DataFrame, out: Path) -> None:
    """Generate risk distribution bar chart."""
    counts = risk_df["Risk_Category"].value_counts().reindex(["Low", "Medium", "High"])
    colors = [get_risk_color(cat) for cat in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=12)

    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Number of Patients")
    ax.set_title("CKD Risk Distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "risk_distribution_bar.png", dpi=150)
    plt.close(fig)
    logger.info("Saved risk distribution bar chart.")


def _plot_risk_histogram(risk_df: pd.DataFrame, out: Path) -> None:
    """Generate CKD probability histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        risk_df["CKD_Probability"],
        bins=20,
        color="#3498db",
        edgecolor="black",
        alpha=0.8,
    )
    ax.axvline(0.3, color="#f39c12", linestyle="--", linewidth=2, label="Low/Medium (0.3)")
    ax.axvline(0.7, color="#e74c3c", linestyle="--", linewidth=2, label="Medium/High (0.7)")
    ax.set_xlabel("CKD Probability")
    ax.set_ylabel("Number of Patients")
    ax.set_title("CKD Risk Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "risk_score_histogram.png", dpi=150)
    plt.close(fig)
    logger.info("Saved risk score histogram.")
