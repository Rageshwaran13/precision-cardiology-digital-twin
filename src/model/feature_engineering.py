"""
feature_engineering.py
-----------------------
Functions to build the feature matrix from the merged cardiology dataset
and extract feature importance from a trained model.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns that must never appear in the feature matrix
# ---------------------------------------------------------------------------
_DROP_COLS = [
    "patient_id",
    "timestamp",
    "event_flag",
    "ischemic_risk_score",
    # Any derivative target columns that may appear
    "event_onset_time",
    "time_to_event_h",
]

# Columns to one-hot encode
_CATEGORICAL_COLS = ["sex", "sleep_stage"]


def build_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build a cleaned, imputed feature matrix from the merged DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataset containing both static EHR features and
        dynamic wearable rolling-window features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix ready for modelling.
    y : pd.Series
        Binary target (event_flag): 1 = within 24 h pre-event window.
    feature_names : list[str]
        Ordered list of feature column names (matches X.columns).
    """
    df = df.copy()

    # ---- Extract target ------------------------------------------------
    if "event_flag" not in df.columns:
        raise ValueError("'event_flag' column not found in DataFrame.")
    y = df["event_flag"].astype(int)

    # ---- Drop non-feature columns that exist in the DataFrame ----------
    drop_existing = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)

    # ---- One-hot encode categorical columns ----------------------------
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # ---- Convert all remaining object / bool columns ------------------
    for col in df.select_dtypes(include=["object", "bool"]).columns:
        try:
            df[col] = df[col].astype(float)
        except (ValueError, TypeError):
            # If it can't be cast to float drop it with a warning
            logger.warning("Dropping unconvertible column: %s", col)
            df = df.drop(columns=[col])

    # ---- Median imputation for missing values --------------------------
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(df)
    X = pd.DataFrame(X_arr, columns=df.columns, index=df.index)

    feature_names = list(X.columns)
    logger.info("Feature matrix built: %d rows × %d features", len(X), len(feature_names))

    return X, y, feature_names


def get_feature_importance_df(
    model,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Return a DataFrame of features sorted by importance (descending).

    Works with any sklearn-compatible estimator that exposes
    ``feature_importances_`` (tree-based models, XGBoost wrappers, etc.).

    Parameters
    ----------
    model :
        Trained classifier with a ``feature_importances_`` attribute.
    feature_names : list[str]
        Ordered list of feature names matching the training feature matrix.

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'importance'], sorted by importance descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The model does not expose 'feature_importances_'. "
            "Ensure it is a tree-based estimator."
        )

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Mismatch: model has {len(importances)} importances "
            f"but {len(feature_names)} feature names were provided."
        )

    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return imp_df


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Tiny synthetic test
    rng = np.random.default_rng(42)
    n = 50
    dummy = pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(n)],
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "age": rng.integers(40, 80, n),
            "sex": rng.choice(["M", "F"], n),
            "bmi": rng.uniform(20, 35, n),
            "event_flag": rng.integers(0, 2, n),
            "hrv_rmssd_6h_mean": rng.uniform(20, 60, n),
            "spo2_pct_6h_min": rng.uniform(90, 99, n),
        }
    )
    X, y, names = build_feature_matrix(dummy)
    print("X shape:", X.shape)
    print("Feature names:", names)
    print("y distribution:\n", y.value_counts())
