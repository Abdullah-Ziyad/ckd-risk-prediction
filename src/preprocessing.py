"""
Module 2: Preprocessing
Encodes categorical features, scales numerical features, splits data
into train/validation/test (70/20/10), and applies SMOTE to handle class imbalance.
"""

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# Binary Yes/No columns
BINARY_YES_NO_COLUMNS = [
    "have_you_ever_been_diagnosed_with_diabetes",
    "do_you_have_high_blood_pressure_hypertension",
    "do_you_notice_foamy_urine",
    "do_you_take_extra_salt_with_your_food",
]

# Numerical features to scale
NUMERICAL_FEATURES = ["age", "hemoglobin"]

# Features excluded from prediction to avoid data leakage / circularity:
# - egfr: CKD is clinically defined by eGFR thresholds (direct leakage)
# - ckd_stage: directly derived from eGFR
# - serum_creatinine: eGFR is computed from serum creatinine via CKD-EPI;
#   including it inflates accuracy beyond the realistic range
EXCLUDE_FROM_PREDICTION = ["egfr", "ckd_stage", "serum_creatinine"]

# Target column
TARGET = "ckd_diagnosis"

# Random seed for reproducibility
RANDOM_STATE = 42


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns to numerical values.

    Args:
        df: Raw cleaned DataFrame.

    Returns:
        DataFrame with all features encoded numerically.
    """
    df = df.copy()

    # Encode Yes/No columns -> 1/0
    for col in BINARY_YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
            if df[col].isnull().any():
                logger.warning("Unmapped values in column '%s', filling with 0.", col)
                df[col] = df[col].fillna(0).astype(int)

    # Encode gender: Male=1, Female=0
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
        if df["gender"].isnull().any():
            df["gender"] = df["gender"].fillna(0).astype(int)

    # Encode living_area if present: Urban=1, Rural=0
    if "living_area" in df.columns:
        df["living_area"] = df["living_area"].map({"Urban": 1, "Rural": 0})
        if df["living_area"].isnull().any():
            df["living_area"] = df["living_area"].fillna(0).astype(int)

    logger.info("Categorical encoding complete.")
    return df


def preprocess(
    df: pd.DataFrame,
    apply_smote: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series, StandardScaler, List[str]]:
    """Full preprocessing pipeline with 70/20/10 stratified split.

    eGFR and ckd_stage are excluded from prediction features because CKD is
    defined by eGFR thresholds — including them would constitute circular
    reasoning / data leakage. eGFR is used separately for CKD staging.

    Args:
        df: Cleaned DataFrame (from data_loader).
        apply_smote: Whether to apply SMOTE on training data.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
    """
    # Encode all categorical features
    df_encoded = encode_features(df)

    # Separate features and target — EXCLUDE eGFR and ckd_stage
    feature_cols = [
        c for c in df_encoded.columns
        if c != TARGET and c not in EXCLUDE_FROM_PREDICTION
    ]
    X = df_encoded[feature_cols]
    y = df_encoded[TARGET]
    feature_names = list(X.columns)

    logger.info("Prediction features (%d): %s", len(feature_names), feature_names)
    logger.info("Excluded from prediction (used for staging): %s", EXCLUDE_FROM_PREDICTION)
    logger.info("Target distribution:\n%s", y.value_counts().to_string())

    # ── Stratified 70/20/10 split ──
    # Step 1: Split off 10% test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, stratify=y
    )
    # Step 2: Split remaining 90% into 70% train and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=RANDOM_STATE, stratify=y_temp
    )

    logger.info(
        "Stratified split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(X_train), len(X_train) / len(X) * 100,
        len(X_val), len(X_val) / len(X) * 100,
        len(X_test), len(X_test) / len(X) * 100,
    )

    # Scale numerical features only (fit on train, transform all three sets)
    scaler = StandardScaler()
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    logger.info("Scaled numerical features: %s", num_cols)

    # Apply SMOTE on training data only
    if apply_smote:
        logger.info(
            "Before SMOTE — Train class distribution:\n%s",
            y_train.value_counts().to_string(),
        )
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.6)
        X_train_values, y_train_values = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_values, columns=feature_names)
        y_train = pd.Series(y_train_values, name=TARGET)
        logger.info(
            "After SMOTE — Train class distribution:\n%s",
            y_train.value_counts().to_string(),
        )

    return (
        X_train.values,
        X_val.values,
        X_test.values,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_names,
    )
