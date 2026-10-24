"""
merge_datasets.py
-----------------
Merges the EHR static features with the wearable time-series data, then
engineers rolling window features and computes an ``ischemic_risk_score``
for each patient-hour observation.

Rolling features (6-hour look-back window):
  • hrv_rmssd_6h_mean, hrv_rmssd_6h_std
  • heart_rate_6h_mean, heart_rate_6h_max
  • sleep_fragmentation_6h_mean
  • spo2_pct_6h_min

Ischemic Risk Score (0.0 – 1.0):
  Composite of static risk factors (age, comorbidities, LVEF) and
  dynamic anomaly indicators (rolling HRV depression, tachycardia, SpO2 dip,
  sleep fragmentation, event_flag).

Output: data/processed/merged_dataset.csv
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EHR_PATH = os.path.join(ROOT_DIR, "data", "raw", "ehr_data.csv")
WEARABLE_PATH = os.path.join(ROOT_DIR, "data", "raw", "wearable_data.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "processed", "merged_dataset.csv")

ROLLING_WINDOW = 6  # hours


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data(ehr_path: str = EHR_PATH, wearable_path: str = WEARABLE_PATH):
    """
    Load EHR and wearable CSVs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (ehr_df, wearable_df)
    """
    print(f"[Merge] Loading EHR from      : {ehr_path}")
    ehr_df = pd.read_csv(ehr_path)

    print(f"[Merge] Loading wearable from : {wearable_path}")
    wearable_df = pd.read_csv(wearable_path, parse_dates=["timestamp"])

    print(f"  EHR shape      : {ehr_df.shape}")
    print(f"  Wearable shape : {wearable_df.shape}")
    return ehr_df, wearable_df


# ── Step 2: Join EHR static features ─────────────────────────────────────────

def join_ehr(wearable_df: pd.DataFrame, ehr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join EHR static features onto every wearable row by ``patient_id``.

    Parameters
    ----------
    wearable_df : pd.DataFrame
    ehr_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all wearable columns + EHR static columns.
    """
    merged = wearable_df.merge(ehr_df, on="patient_id", how="left")
    print(f"[Merge] After EHR join: {merged.shape}")
    return merged


# ── Step 3: Compute rolling features ─────────────────────────────────────────

def compute_rolling_features(merged: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Compute rolling window aggregates per patient over the previous *window*
    hours (inclusive of current row → ``min_periods=1``).

    Features added:
      • hrv_rmssd_6h_mean, hrv_rmssd_6h_std
      • heart_rate_6h_mean, heart_rate_6h_max
      • sleep_fragmentation_6h_mean
      • spo2_pct_6h_min

    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame sorted by patient_id, timestamp.
    window : int
        Rolling window size in hours.

    Returns
    -------
    pd.DataFrame
        DataFrame with new rolling columns appended.
    """
    print(f"[Merge] Computing {window}-hour rolling features …")
    merged = merged.sort_values(["patient_id", "timestamp"]).copy()

    rolling_specs = {
        "hrv_rmssd_6h_mean": ("hrv_rmssd", "mean"),
        "hrv_rmssd_6h_std":  ("hrv_rmssd", "std"),
        "heart_rate_6h_mean": ("heart_rate", "mean"),
        "heart_rate_6h_max":  ("heart_rate", "max"),
        "sleep_fragmentation_6h_mean": ("sleep_fragmentation_index", "mean"),
        "spo2_pct_6h_min":    ("spo2_pct", "min"),
    }

    grouped = merged.groupby("patient_id", sort=False)

    for new_col, (src_col, agg_fn) in rolling_specs.items():
        rolled = grouped[src_col].transform(
            lambda s: s.rolling(window=window, min_periods=1).agg(agg_fn)
        )
        merged[new_col] = rolled.values

    # Fill any NaN introduced by std on single-element windows
    merged["hrv_rmssd_6h_std"] = merged["hrv_rmssd_6h_std"].fillna(0.0)

    print(f"[Merge] Rolling features computed. Shape: {merged.shape}")
    return merged


# ── Step 4: Compute ischemic risk score ───────────────────────────────────────

def compute_risk_score(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite ``ischemic_risk_score`` in [0.0, 1.0] for every row.

    Formula
    -------
    The score is a weighted sigmoid of a raw risk index built from:

    Static factors (weight 0.45):
      • Normalised age (0-1 over range 40-85)           × 0.15
      • Presence of hypertension                        × 0.07
      • Presence of hyperlipidemia                      × 0.06
      • Presence of type2_diabetes                      × 0.07
      • Normalised LVEF depression (1 - lvef/75)        × 0.10

    Dynamic / wearable factors (weight 0.55):
      • Normalised HRV depression (1 - rmssd_mean/80)   × 0.15
      • HRV instability (rmssd_std / 40)                × 0.05
      • Heart rate elevation (hr_mean - 60) / 70        × 0.10
      • Sleep fragmentation 6h mean                     × 0.10
      • SpO2 deficit (1 - spo2_min/100)                 × 0.10
      • event_flag                                      × 0.05

    The raw index is passed through a soft clamp and normalised to [0,1].

    Parameters
    ----------
    merged : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with ``ischemic_risk_score`` column added.
    """
    print("[Merge] Computing ischemic_risk_score …")

    # Static components
    age_norm = (merged["age"] - 40) / 45.0
    lvef_depr = 1.0 - (merged["lvef_pct"] / 75.0)

    static_index = (
        0.15 * age_norm.clip(0, 1) +
        0.07 * merged["hypertension"] +
        0.06 * merged["hyperlipidemia"] +
        0.07 * merged["type2_diabetes"] +
        0.10 * lvef_depr.clip(0, 1)
    )  # max ≈ 0.45

    # Dynamic components
    hrv_depr = (1.0 - merged["hrv_rmssd_6h_mean"] / 80.0).clip(0, 1)
    hrv_instab = (merged["hrv_rmssd_6h_std"] / 40.0).clip(0, 1)
    hr_elev = ((merged["heart_rate_6h_mean"] - 60.0) / 70.0).clip(0, 1)
    sfrag = merged["sleep_fragmentation_6h_mean"].clip(0, 1)
    spo2_def = (1.0 - merged["spo2_pct_6h_min"] / 100.0).clip(0, 1)

    dynamic_index = (
        0.15 * hrv_depr +
        0.05 * hrv_instab +
        0.10 * hr_elev +
        0.10 * sfrag +
        0.10 * spo2_def +
        0.05 * merged["event_flag"]
    )  # max ≈ 0.55

    raw_score = (static_index + dynamic_index).clip(0.0, 1.0)

    # Sigmoid stretch to spread the distribution across [0,1]
    def _sigmoid_stretch(x: pd.Series, scale: float = 8.0, shift: float = 0.45) -> pd.Series:
        """Apply sigmoid to spread the score distribution."""
        z = scale * (x - shift)
        return 1.0 / (1.0 + np.exp(-z))

    stretched = _sigmoid_stretch(raw_score)
    # Re-normalise to exactly [0,1]
    mn, mx = stretched.min(), stretched.max()
    ischemic_risk_score = ((stretched - mn) / (mx - mn + 1e-9)).round(4)

    merged["ischemic_risk_score"] = ischemic_risk_score
    print(f"  Risk score — min: {ischemic_risk_score.min():.4f}  "
          f"max: {ischemic_risk_score.max():.4f}  "
          f"mean: {ischemic_risk_score.mean():.4f}")
    return merged


# ── Step 5: Save ──────────────────────────────────────────────────────────────

def save_merged(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    """Save the merged DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"[Merge] Saved {len(df):,} rows x {len(df.columns)} cols "
          f"-> {path}  ({size_mb:.1f} MB)")


def print_summary(df: pd.DataFrame) -> None:
    """Print a brief statistical summary of the merged dataset."""
    print("\n" + "=" * 65)
    print("  Merged Dataset Summary")
    print("=" * 65)
    print(f"  Rows            : {len(df):,}")
    print(f"  Columns         : {len(df.columns)}")
    print(f"  Patients        : {df['patient_id'].nunique():,}")
    print(f"  Event-flag rows : {df['event_flag'].sum():,}")
    print(f"\n  Risk Score Distribution:")
    for label, (lo, hi) in [("LOW", (0.0, 0.3)), ("MODERATE", (0.3, 0.6)),
                              ("HIGH", (0.6, 0.85)), ("CRITICAL", (0.85, 1.01))]:
        cnt = ((df["ischemic_risk_score"] >= lo) & (df["ischemic_risk_score"] < hi)).sum()
        print(f"    {label:<10}: {cnt:>9,}  ({cnt/len(df)*100:.2f}%)")
    print("=" * 65)
    print("\n  Columns in merged dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:>3}. {col}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """Execute the full merge pipeline and return the merged DataFrame."""
    ehr_df, wearable_df = load_data()
    merged = join_ehr(wearable_df, ehr_df)
    merged = compute_rolling_features(merged)
    merged = compute_risk_score(merged)
    return merged


if __name__ == "__main__":
    print("Running EHR + Wearable merge pipeline …\n")
    merged_df = run_pipeline()
    save_merged(merged_df)
    print_summary(merged_df)
