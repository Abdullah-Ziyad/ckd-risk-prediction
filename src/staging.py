"""
Module 7: eGFR & CKD Staging
Maps eGFR values to CKD stages (1-5) and generates stage distribution plots.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# CKD stage definitions
CKD_STAGES = {
    "Stage 1": {"range": (90, float("inf")), "description": "Normal or High"},
    "Stage 2": {"range": (60, 90), "description": "Mildly Decreased"},
    "Stage 3a": {"range": (45, 60), "description": "Mild-Moderate Decrease"},
    "Stage 3b": {"range": (30, 45), "description": "Moderate-Severe Decrease"},
    "Stage 4": {"range": (15, 30), "description": "Severely Decreased"},
    "Stage 5": {"range": (0, 15), "description": "Kidney Failure"},
}


def classify_egfr_stage(egfr: float) -> str:
    """Classify a single eGFR value into a CKD stage based on eGFR ALONE.

    NOTE: Per KDIGO criteria, Stage 1 and Stage 2 also require markers of
    kidney damage (e.g. urine abnormalities like proteinuria/foamy urine).
    For full KDIGO classification use classify_ckd_stage_kdigo() instead.

    Args:
        egfr: Estimated glomerular filtration rate.

    Returns:
        CKD stage string (e.g., "Stage 1").
    """
    if egfr >= 90:
        return "Stage 1"
    elif egfr >= 60:
        return "Stage 2"
    elif egfr >= 45:
        return "Stage 3a"
    elif egfr >= 30:
        return "Stage 3b"
    elif egfr >= 15:
        return "Stage 4"
    else:
        return "Stage 5"


def classify_ckd_stage_kdigo(egfr: float, has_urine_abnormality: bool) -> str:
    """Classify CKD stage following full KDIGO criteria.

    Stage 1 and Stage 2 require BOTH:
      - eGFR in range (>=90 for S1, 60-89 for S2)
      - markers of kidney damage (urine abnormality, e.g. proteinuria/foamy urine)

    Stages 3a, 3b, 4, 5 are determined by eGFR alone.

    Args:
        egfr: Estimated glomerular filtration rate (mL/min/1.73m^2).
        has_urine_abnormality: True if patient has urine markers (e.g. foamy urine).

    Returns:
        CKD stage string. Returns "No CKD (Normal eGFR)" or
        "No CKD (Mild eGFR decrease, no urine markers)" when eGFR is in
        Stage 1/2 range but no urine abnormality is present.
    """
    if egfr >= 90:
        if has_urine_abnormality:
            return "Stage 1"
        return "No CKD (Normal eGFR, no urine markers)"
    elif egfr >= 60:
        if has_urine_abnormality:
            return "Stage 2"
        return "No CKD (Mild eGFR decrease, no urine markers)"
    elif egfr >= 45:
        return "Stage 3a"
    elif egfr >= 30:
        return "Stage 3b"
    elif egfr >= 15:
        return "Stage 4"
    else:
        return "Stage 5"


def get_stage_description(stage: str) -> str:
    """Get the clinical description for a CKD stage."""
    if stage and stage.startswith("No CKD"):
        if "Normal eGFR" in stage:
            return "Normal kidney function — no CKD criteria met"
        return "Mild eGFR decrease — but no urine markers, KDIGO criteria for CKD not met"
    return CKD_STAGES.get(stage, {}).get("description", "Unknown")


def get_stage_color(stage: str) -> str:
    """Return color for a CKD stage (for visualizations)."""
    colors = {
        "Stage 1": "#2ecc71",
        "Stage 2": "#27ae60",
        "Stage 3a": "#f1c40f",
        "Stage 3b": "#e67e22",
        "Stage 4": "#e74c3c",
        "Stage 5": "#c0392b",
    }
    return colors.get(stage, "#95a5a6")


def estimate_egfr_ckd_epi(
    serum_creatinine: float, age: float, is_female: bool
) -> float:
    """Estimate eGFR using the CKD-EPI 2021 formula.

    Formula (no race coefficient, matches reference):
      Male   (S.cr <= 0.9): eGFR = 142 * (S.cr/0.9)^(-0.302) * (0.9938)^Age
      Male   (S.cr >  0.9): eGFR = 142 * (S.cr/0.9)^(-1.200) * (0.9938)^Age
      Female (S.cr <= 0.7): eGFR = 142 * (S.cr/0.7)^(-0.241) * (0.9938)^Age
      Female (S.cr >  0.7): eGFR = 142 * (S.cr/0.7)^(-1.200) * (0.9938)^Age

    Args:
        serum_creatinine: Serum creatinine in mg/dL.
        age: Patient age in years.
        is_female: True if the patient is female.

    Returns:
        Estimated GFR in mL/min/1.73m2, rounded to 2 decimal places.
    """
    if is_female:
        kappa = 0.7
        alpha = -0.241
    else:
        kappa = 0.9
        alpha = -0.302

    scr_ratio = serum_creatinine / kappa

    # If S.cr <= kappa: use alpha exponent; else use -1.200
    if serum_creatinine <= kappa:
        exponent = alpha
    else:
        exponent = -1.200

    egfr = 142 * (scr_ratio ** exponent) * (0.9938 ** age)
    return round(egfr, 2)


def compute_staging(
    df: pd.DataFrame, output_dir: str = "outputs"
) -> pd.DataFrame:
    """Compute CKD stages from the dataset.

    Uses the egfr column directly if available, otherwise estimates via CKD-EPI.

    Args:
        df: Cleaned DataFrame (encoded or raw — needs egfr column).
        output_dir: Directory to save plots.

    Returns:
        DataFrame with eGFR values, CKD stages, and descriptions.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    staging_df = pd.DataFrame()

    if "egfr" in df.columns:
        logger.info("Using existing eGFR column for staging.")
        staging_df["eGFR"] = df["egfr"].values
    else:
        logger.info("eGFR column not found — estimating via CKD-EPI formula.")
        egfr_values = []
        for _, row in df.iterrows():
            is_female = (
                row.get("gender") == "Female"
                or row.get("gender") == 0
            )
            egfr = estimate_egfr_ckd_epi(
                serum_creatinine=row["serum_creatinine"],
                age=row["age"],
                is_female=is_female,
            )
            egfr_values.append(egfr)
        staging_df["eGFR"] = egfr_values

    staging_df["CKD_Stage"] = staging_df["eGFR"].apply(classify_egfr_stage)
    staging_df["Stage_Description"] = staging_df["CKD_Stage"].apply(get_stage_description)

    logger.info(
        "CKD Stage distribution:\n%s",
        staging_df["CKD_Stage"].value_counts().to_string(),
    )

    # Save staging data
    staging_df.to_csv(out / "ckd_staging.csv", index=False)

    # Generate plots
    _plot_stage_distribution(staging_df, out)
    _plot_egfr_histogram(staging_df, out)

    return staging_df


def _plot_stage_distribution(staging_df: pd.DataFrame, out: Path) -> None:
    """Generate CKD stage distribution bar chart."""
    stage_order = ["Stage 1", "Stage 2", "Stage 3a", "Stage 3b", "Stage 4", "Stage 5"]
    counts = staging_df["CKD_Stage"].value_counts().reindex(stage_order, fill_value=0)
    colors = [get_stage_color(s) for s in counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("CKD Stage")
    ax.set_ylabel("Number of Patients")
    ax.set_title("CKD Stage Distribution (based on eGFR)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "ckd_stage_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved CKD stage distribution bar chart.")


def _plot_egfr_histogram(staging_df: pd.DataFrame, out: Path) -> None:
    """Generate eGFR distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        staging_df["eGFR"],
        bins=30,
        color="#3498db",
        edgecolor="black",
        alpha=0.8,
    )
    # Stage boundary lines
    boundaries = [
        (90, "Stage 1/2"),
        (60, "Stage 2/3a"),
        (45, "Stage 3a/3b"),
        (30, "Stage 3b/4"),
        (15, "Stage 4/5"),
    ]
    for val, label in boundaries:
        ax.axvline(val, color="red", linestyle="--", alpha=0.7)
        ax.text(val + 1, ax.get_ylim()[1] * 0.9, label, fontsize=8, rotation=90)

    ax.set_xlabel("eGFR (mL/min/1.73m²)")
    ax.set_ylabel("Number of Patients")
    ax.set_title("eGFR Distribution")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "egfr_distribution_histogram.png", dpi=150)
    plt.close(fig)
    logger.info("Saved eGFR distribution histogram.")
