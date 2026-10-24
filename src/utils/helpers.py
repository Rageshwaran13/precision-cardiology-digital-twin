"""
helpers.py
----------
Shared utility functions for the Precision Cardiology Digital Twin project.

Provides data-loading helpers, patient profiling, risk label computation,
and formatted risk-banner generation used across notebooks and application code.
"""

import os
from typing import Any

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EHR_PATH = os.path.join(_ROOT_DIR, "data", "raw", "ehr_data.csv")
WEARABLE_PATH = os.path.join(_ROOT_DIR, "data", "raw", "wearable_data.csv")
MERGED_PATH = os.path.join(_ROOT_DIR, "data", "processed", "merged_dataset.csv")

# ── Risk thresholds ───────────────────────────────────────────────────────────
_RISK_THRESHOLDS = [
    (0.00, 0.30, "LOW"),
    (0.30, 0.60, "MODERATE"),
    (0.60, 0.85, "HIGH"),
    (0.85, 1.01, "CRITICAL"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_ehr(path: str = EHR_PATH) -> pd.DataFrame:
    """
    Load the EHR dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Path to ehr_data.csv.  Defaults to the standard project location.

    Returns
    -------
    pd.DataFrame
        EHR DataFrame with one row per patient.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"EHR file not found: {path}\n"
            "Run `src/data/generate_ehr.py` first."
        )
    df = pd.read_csv(path)
    return df


def load_wearable(path: str = WEARABLE_PATH) -> pd.DataFrame:
    """
    Load the wearable time-series dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Path to wearable_data.csv.

    Returns
    -------
    pd.DataFrame
        Wearable DataFrame with timestamps parsed as ``datetime64``.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Wearable file not found: {path}\n"
            "Run `src/data/generate_wearable.py` first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_merged(path: str = MERGED_PATH) -> pd.DataFrame:
    """
    Load the merged (EHR + wearable) dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Path to merged_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with timestamps parsed and all engineered features.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Merged file not found: {path}\n"
            "Run `src/data/merge_datasets.py` first."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Patient Profiling
# ══════════════════════════════════════════════════════════════════════════════

def get_patient_profile(patient_id: str) -> dict[str, Any]:
    """
    Return a combined profile dict containing EHR static features and the
    most recent wearable snapshot (latest timestamp row) for *patient_id*.

    Parameters
    ----------
    patient_id : str
        Patient identifier, e.g. ``'PAT-0001'``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
          • ``'ehr'`` – dict of EHR features
          • ``'latest_wearable'`` – dict of the most recent wearable reading
          • ``'latest_timestamp'`` – the timestamp of the most recent reading
          • ``'ischemic_risk_score'`` – float (from merged dataset, latest row)
          • ``'risk_label'`` – str ('LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL')

    Raises
    ------
    KeyError
        If *patient_id* is not found in the dataset.
    """
    ehr_df = load_ehr()
    ehr_row = ehr_df[ehr_df["patient_id"] == patient_id]
    if ehr_row.empty:
        raise KeyError(f"Patient '{patient_id}' not found in EHR data.")

    merged_df = load_merged()
    patient_merged = merged_df[merged_df["patient_id"] == patient_id]
    if patient_merged.empty:
        raise KeyError(f"Patient '{patient_id}' not found in merged data.")

    latest_row = patient_merged.sort_values("timestamp").iloc[-1]

    # EHR columns
    ehr_cols = [
        "patient_id", "age", "sex", "bmi", "hypertension", "hyperlipidemia",
        "type2_diabetes", "smoking_index", "prev_ischemic_episodes",
        "lvef_pct", "resting_hr", "resting_ecg_qt_ms", "resting_ecg_pr_ms",
        "systolic_bp", "diastolic_bp",
    ]
    ehr_dict = ehr_row.iloc[0][ehr_cols].to_dict()

    # Wearable columns in the latest row
    wearable_cols = [
        "timestamp", "hrv_rmssd", "hrv_sdnn", "heart_rate", "sleep_stage",
        "sleep_fragmentation_index", "active_energy_kcal", "step_frequency",
        "spo2_pct", "skin_temp_celsius", "event_flag",
        "hrv_rmssd_6h_mean", "hrv_rmssd_6h_std",
        "heart_rate_6h_mean", "heart_rate_6h_max",
        "sleep_fragmentation_6h_mean", "spo2_pct_6h_min",
    ]
    existing_wearable_cols = [c for c in wearable_cols if c in latest_row.index]
    wearable_dict = latest_row[existing_wearable_cols].to_dict()

    score = float(latest_row["ischemic_risk_score"])
    label = compute_risk_label(latest_row.to_dict())

    return {
        "ehr": ehr_dict,
        "latest_wearable": wearable_dict,
        "latest_timestamp": latest_row["timestamp"],
        "ischemic_risk_score": round(score, 4),
        "risk_label": label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Risk Utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_label(row: dict[str, Any]) -> str:
    """
    Map an ``ischemic_risk_score`` to a categorical risk label.

    Thresholds:
      • [0.00, 0.30) → ``'LOW'``
      • [0.30, 0.60) → ``'MODERATE'``
      • [0.60, 0.85) → ``'HIGH'``
      • [0.85, 1.00] → ``'CRITICAL'``

    Parameters
    ----------
    row : dict
        Must contain the key ``'ischemic_risk_score'`` (float 0.0 – 1.0).
        Alternatively, pass a pandas Series or any mapping that supports
        subscript access.

    Returns
    -------
    str
        One of ``'LOW'``, ``'MODERATE'``, ``'HIGH'``, ``'CRITICAL'``.
    """
    score = float(row["ischemic_risk_score"])
    for lo, hi, label in _RISK_THRESHOLDS:
        if lo <= score < hi:
            return label
    return "CRITICAL"  # score == 1.0 edge case


def format_risk_banner(
    patient_id: str,
    score: float,
    window_hours: int,
) -> str:
    """
    Format a human-readable risk alert banner string.

    Parameters
    ----------
    patient_id : str
        Patient identifier, e.g. ``'PAT-0001'``.
    score : float
        Ischemic risk score in [0.0, 1.0].
    window_hours : int
        Prediction horizon in hours (e.g. 24 means the risk is for the
        next 24-hour window).

    Returns
    -------
    str
        Formatted banner, e.g.:
        ``"[PAT-0042] Acute Ischemic Anomaly Risk: 84% [Window: 18-24h]"``

    Examples
    --------
    >>> format_risk_banner("PAT-0042", 0.84, 24)
    '[PAT-0042] Acute Ischemic Anomaly Risk: 84% [Window: 18-24h]'
    """
    pct = int(round(score * 100))
    lo = max(0, window_hours - 6)
    hi = window_hours
    label = compute_risk_label({"ischemic_risk_score": score})
    return (
        f"[{patient_id}] Acute Ischemic Anomaly Risk: {pct}% "
        f"[Window: {lo}-{hi}h] | Severity: {label}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading datasets …")
    ehr = load_ehr()
    wearable = load_wearable()
    merged = load_merged()

    print(f"  EHR      : {ehr.shape}")
    print(f"  Wearable : {wearable.shape}")
    print(f"  Merged   : {merged.shape}")

    sample_id = ehr["patient_id"].iloc[0]
    print(f"\nFetching profile for {sample_id} …")
    profile = get_patient_profile(sample_id)
    print(f"  Risk score : {profile['ischemic_risk_score']}")
    print(f"  Risk label : {profile['risk_label']}")

    print("\nRisk label examples:")
    for score in [0.15, 0.45, 0.72, 0.91]:
        label = compute_risk_label({"ischemic_risk_score": score})
        banner = format_risk_banner("PAT-TEST", score, 24)
        print(f"  score={score:.2f}  label={label}  banner='{banner}'")
