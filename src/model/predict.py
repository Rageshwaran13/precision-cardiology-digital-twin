"""
predict.py
----------
Real-time inference module for the Ischemic Cardiac Event Risk Predictor.

Usage
-----
    python src/model/predict.py

Exports
-------
    load_model()                 → (model, feature_names)
    predict_patient_risk(pid)    → dict
    predict_all_patients()       → pd.DataFrame
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _ROOT / "data" / "processed" / "merged_dataset.csv"
_MODELS_DIR = _ROOT / "models"
_MODEL_PATH = _MODELS_DIR / "ischemic_risk_model.pkl"
_FEATURE_NAMES_PATH = _MODELS_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# Risk thresholds  (prob → label)
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    "CRITICAL": 0.85,
    "HIGH":     0.60,
    "MODERATE": 0.30,
}


def _risk_label(prob: float) -> str:
    if prob >= _THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if prob >= _THRESHOLDS["HIGH"]:
        return "HIGH"
    if prob >= _THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model() -> Tuple:
    """
    Load the trained model and feature names from models/.

    Returns
    -------
    model : trained sklearn-compatible classifier
    feature_names : list[str]
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. "
            "Run 'python src/model/train.py' first."
        )
    if not _FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature names not found at {_FEATURE_NAMES_PATH}."
        )

    model = joblib.load(_MODEL_PATH)
    with open(_FEATURE_NAMES_PATH) as f:
        feature_names: List[str] = json.load(f)

    logger.info(
        "Model loaded: %s | %d features",
        type(model).__name__,
        len(feature_names),
    )
    return model, feature_names


# ---------------------------------------------------------------------------
# Internal helper: build features for a single patient slice
# ---------------------------------------------------------------------------

def _build_patient_features(
    patient_df: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Build a single-row (or multi-row) feature DataFrame for a patient slice.
    Applies the same transformations as build_feature_matrix but WITHOUT
    re-fitting an imputer — uses feature_names as the reference schema.
    """
    # Add project root to sys.path so feature_engineering can be imported
    sys.path.insert(0, str(_ROOT / "src"))
    from model.feature_engineering import build_feature_matrix  # noqa: PLC0415

    X, _, _ = build_feature_matrix(patient_df)

    # Align columns to training schema (fill zeros for unseen one-hot cols)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    return X


# ---------------------------------------------------------------------------
# Top-risk-factors proxy (feature * importance)
# ---------------------------------------------------------------------------

def _top_risk_factors(
    X_row: pd.DataFrame,
    model,
    feature_names: List[str],
    top_n: int = 5,
) -> List[str]:
    """
    Return top_n feature names by |feature_value × feature_importance|
    as a SHAP-like proxy.
    """
    if not hasattr(model, "feature_importances_"):
        return feature_names[:top_n]

    importances = model.feature_importances_
    values = X_row.values.flatten()

    # Normalise values to [0, 1] to make the product comparable
    v_min, v_max = values.min(), values.max()
    v_range = v_max - v_min if v_max != v_min else 1.0
    norm_values = (values - v_min) / v_range

    contributions = np.abs(norm_values * importances)
    top_idx = np.argsort(contributions)[::-1][:top_n]
    return [feature_names[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_patient_risk(
    patient_id: str,
    df_merged: Optional[pd.DataFrame] = None,
    model=None,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Predict ischemic event risk for a single patient.

    Parameters
    ----------
    patient_id : str
    df_merged  : optional pre-loaded DataFrame (loaded from CSV if None)
    model      : optional pre-loaded model (loaded from disk if None)
    feature_names : optional list (loaded from disk if None)

    Returns
    -------
    dict with keys:
        patient_id, risk_score, risk_label, window_hours,
        top_risk_factors, timestamp
    """
    # -- Load data --------------------------------------------------------
    if df_merged is None:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(
                f"merged_dataset.csv not found at {_DATA_PATH}."
            )
        df_merged = pd.read_csv(_DATA_PATH, parse_dates=["timestamp"])

    # -- Load model -------------------------------------------------------
    if model is None or feature_names is None:
        model, feature_names = load_model()

    # -- Filter patient ---------------------------------------------------
    patient_df = df_merged[df_merged["patient_id"] == patient_id].copy()
    if patient_df.empty:
        raise ValueError(f"Patient '{patient_id}' not found in dataset.")

    # Sort by time and take the most recent 6 rows
    if "timestamp" in patient_df.columns:
        patient_df = patient_df.sort_values("timestamp")
    recent_df = patient_df.tail(6).reset_index(drop=True)

    # -- Build features ---------------------------------------------------
    X = _build_patient_features(recent_df, feature_names)

    # -- Predict (use the LAST row for the point estimate) ----------------
    X_latest = X.tail(1)
    risk_prob = float(model.predict_proba(X_latest)[0, 1])

    # -- Timestamp of the most recent record ------------------------------
    ts = (
        str(recent_df["timestamp"].iloc[-1])
        if "timestamp" in recent_df.columns
        else "N/A"
    )

    # -- Top risk factors -------------------------------------------------
    top_factors = _top_risk_factors(X_latest, model, feature_names, top_n=5)

    return {
        "patient_id": patient_id,
        "risk_score": round(risk_prob, 4),
        "risk_label": _risk_label(risk_prob),
        "window_hours": 24,
        "top_risk_factors": top_factors,
        "timestamp": ts,
    }


def predict_all_patients(
    df_merged: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with the latest risk snapshot for every patient.

    Returns
    -------
    pd.DataFrame
        Columns: patient_id, risk_score, risk_label, window_hours,
                 top_risk_factors, timestamp
    """
    if df_merged is None:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(
                f"merged_dataset.csv not found at {_DATA_PATH}."
            )
        df_merged = pd.read_csv(_DATA_PATH, parse_dates=["timestamp"])

    model, feature_names = load_model()

    patient_ids = df_merged["patient_id"].unique()
    logger.info("Running inference for %d patients …", len(patient_ids))

    results = []
    for pid in patient_ids:
        try:
            result = predict_patient_risk(
                pid,
                df_merged=df_merged,
                model=model,
                feature_names=feature_names,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Skipping patient %s: %s", pid, exc)

    out_df = pd.DataFrame(results).sort_values("risk_score", ascending=False).reset_index(drop=True)
    return out_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Try loading the model -------------------------------------------
    try:
        model, feature_names = load_model()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # -- Load data -------------------------------------------------------
    if _DATA_PATH.exists():
        df = pd.read_csv(_DATA_PATH, parse_dates=["timestamp"])
        logger.info("Loaded %d rows from %s", len(df), _DATA_PATH)
    else:
        logger.warning("merged_dataset.csv not found — generating synthetic data for demo.")
        # Inline minimal synthetic data (avoids circular import)
        sys.path.insert(0, str(_ROOT / "src"))
        from model.train import _generate_synthetic_dataset  # noqa: PLC0415
        df = _generate_synthetic_dataset(n_patients=20, rows_per_patient=24)

    # -- Show 3 example predictions -------------------------------------
    sample_patients = df["patient_id"].unique()[:3]

    print("\n" + "=" * 60)
    print("EXAMPLE PATIENT PREDICTIONS")
    print("=" * 60)
    for pid in sample_patients:
        result = predict_patient_risk(
            pid,
            df_merged=df,
            model=model,
            feature_names=feature_names,
        )
        print(f"\nPatient : {result['patient_id']}")
        print(f"  Risk Score  : {result['risk_score']:.4f}")
        print(f"  Risk Label  : {result['risk_label']}")
        print(f"  Window (h)  : {result['window_hours']}")
        print(f"  Timestamp   : {result['timestamp']}")
        print(f"  Top Factors : {', '.join(result['top_risk_factors'])}")

    # -- Summary table for all patients ---------------------------------
    print("\n" + "=" * 60)
    print("ALL-PATIENT RISK SUMMARY (top 10 by risk score)")
    print("=" * 60)
    summary_df = predict_all_patients(df_merged=df)
    cols = ["patient_id", "risk_score", "risk_label", "timestamp"]
    print(summary_df[cols].head(10).to_string(index=False))
    print("=" * 60)
