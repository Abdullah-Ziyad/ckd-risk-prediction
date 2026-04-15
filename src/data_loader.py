"""
Module 1: Data Input
Loads and performs initial inspection of the CKD dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns to drop from the raw dataset
DROP_COLUMNS = [
    "i_agree_that_this_data_may_be_used_for_research_purposes",
    "timestamp",
]

# Short display names for plotting and reports
DISPLAY_NAMES = {
    "Age": "Age",
    "Gender": "Gender",
    "have_you_ever_been_diagnosed_with_diabetes": "Diabetes",
    "do_you_have_high_blood_pressure_hypertension": "Hypertension",
    "do_you_notice_foamy_urine": "Foamy Urine",
    "do_you_take_extra_salt_with_your_food": "Extra Salt Intake",
    "serum_creatinine": "Serum Creatinine",
    "egfr": "eGFR",
    "hemoglobin": "Hemoglobin",
    "ckd_diagnosis": "CKD Diagnosis",
    "ckd_stage": "CKD Stage",
    # Legacy columns (for backward compatibility)
    "age": "Age",
    "gender": "Gender",
    "living_area": "Living Area",
    "does_anyone_in_your_family_have_chronic_kidney_disease_ckd": "Family CKD History",
    "have_you_experienced_swelling_in_feet_or_face_recently": "Swelling",
    "do_you_have_very_low_or_reduced_urine_output": "Low Urine Output",
    "do_you_smoke_or_take_guljarda": "Smoking/Guljarda",
    "do_you_use_painkillers_regularly_like_napa_voltaren_fenadin": "Regular Painkillers",
}


def load_data(file_path: str = "data/final_verified_ckd_dataset_fixed.xlsx") -> pd.DataFrame:
    """Load the CKD dataset from an Excel file and perform initial cleaning.

    Args:
        file_path: Path to the Excel file.

    Returns:
        Cleaned DataFrame with irrelevant columns dropped.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_excel(path)

    # Normalize column names to lowercase for consistency
    df.columns = [c.strip().lower() for c in df.columns]

    # Display dataset info
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Data types:\n%s", df.dtypes.to_string())
    logger.info("First 5 rows:\n%s", df.head().to_string())

    # Handle null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("Null values found:\n%s", null_counts[null_counts > 0].to_string())
        # Fill hemoglobin nulls with median
        if "hemoglobin" in df.columns and df["hemoglobin"].isnull().any():
            median_hb = df["hemoglobin"].median()
            df["hemoglobin"] = df["hemoglobin"].fillna(median_hb)
            logger.info("Filled hemoglobin nulls with median: %.2f", median_hb)
    else:
        logger.info("No null values found in the dataset.")

    # Drop irrelevant columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    # Validate that target column exists
    if "ckd_diagnosis" not in df.columns:
        raise ValueError("Target column 'ckd_diagnosis' not found in the dataset.")

    logger.info("Cleaned dataset shape: %s", df.shape)
    return df


def get_display_name(col: str) -> str:
    """Get a short display name for a column."""
    return DISPLAY_NAMES.get(col, col.replace("_", " ").title())


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics of the dataset.

    Args:
        df: The cleaned DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "total_patients": len(df),
        "total_features": len(df.columns) - 1,  # exclude target
        "ckd_positive": int((df["ckd_diagnosis"] == 1).sum()),
        "ckd_negative": int((df["ckd_diagnosis"] == 0).sum()),
        "ckd_percentage": round((df["ckd_diagnosis"] == 1).mean() * 100, 1),
        "avg_age": round(df["age"].mean(), 1),
        "avg_serum_creatinine": round(df["serum_creatinine"].mean(), 2),
        "avg_egfr": round(df["egfr"].mean(), 1) if "egfr" in df.columns else "N/A",
        "avg_hemoglobin": round(df["hemoglobin"].mean(), 1),
        "null_count": int(df.isnull().sum().sum()),
    }
    return summary
