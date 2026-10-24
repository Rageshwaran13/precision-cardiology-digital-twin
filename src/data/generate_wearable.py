"""
generate_wearable.py
--------------------
Generates synthetic wearable / IoT time-series data for 500 patients.
Each patient receives 30 days × 24 hourly readings = 720 rows.

Approximately 20 % of patients (high-risk cohort) have realistic anomalous
patterns injected in the final 24-hour window:
  • Degrading HRV (lower RMSSD / SDNN)
  • Rising heart rate
  • Fragmented sleep (higher sleep_fragmentation_index, fewer deep/REM stages)
  • Mild SpO2 dip

A binary column ``event_flag`` marks rows in those anomaly windows (= 1).

Output: data/raw/wearable_data.csv
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EHR_PATH = os.path.join(ROOT_DIR, "data", "raw", "ehr_data.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "raw", "wearable_data.csv")

N_DAYS = 30
HOURS_PER_DAY = 24
ROWS_PER_PATIENT = N_DAYS * HOURS_PER_DAY  # 720
ANOMALY_FRACTION = 0.20

# Reference start timestamp: past 30 days up to current local date
NOW_UTC = datetime.now()
EPOCH = NOW_UTC - timedelta(days=N_DAYS)

# Sleep stage categories
SLEEP_STAGES = ["awake", "light", "deep", "rem"]
# Hourly distribution weights (0-23 h) – realistic; peak awake during day
_AWAKE_HOURS = set(range(6, 23))  # simplified

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_sleep_stage(hour: int, n: int, fragmented: bool = False) -> np.ndarray:
    """
    Sample sleep stages given hour-of-day.

    Parameters
    ----------
    hour : int
        Hour of day (0-23).
    n : int
        Number of samples.
    fragmented : bool
        If True, use weights that favour awake/light (fragmented sleep).

    Returns
    -------
    np.ndarray of str
        Sampled sleep-stage labels.
    """
    if hour in _AWAKE_HOURS:
        weights = [0.85, 0.10, 0.02, 0.03]
    else:
        if fragmented:
            weights = [0.40, 0.38, 0.10, 0.12]
        else:
            weights = [0.10, 0.35, 0.30, 0.25]
    return np.random.choice(SLEEP_STAGES, size=n, p=weights)


def generate_patient_wearable(
    patient_id: str,
    is_high_risk: bool,
) -> pd.DataFrame:
    """
    Generate wearable time-series for a single patient.

    Parameters
    ----------
    patient_id : str
        EHR patient identifier, e.g. ``'PAT-0001'``.
    is_high_risk : bool
        If True, anomalous patterns are injected in the last 24-hour window.

    Returns
    -------
    pd.DataFrame
        720-row DataFrame with wearable sensor columns.
    """
    n = ROWS_PER_PATIENT
    timestamps = [EPOCH + timedelta(hours=h) for h in range(n)]
    hours_of_day = np.array([ts.hour for ts in timestamps])

    # ── Baseline signals ──────────────────────────────────────────────────────
    # HRV RMSSD: higher at night/rest, lower during activity
    hrv_rmssd_base = 45 + 15 * np.sin(np.linspace(0, 2 * np.pi, HOURS_PER_DAY))
    hrv_rmssd_day = np.tile(hrv_rmssd_base, N_DAYS)
    hrv_rmssd = hrv_rmssd_day + np.random.normal(0, 6, n)
    hrv_rmssd = np.clip(hrv_rmssd, 10.0, 80.0).round(2)

    # HRV SDNN: correlated with RMSSD but slightly larger
    hrv_sdnn = hrv_rmssd * 1.3 + np.random.normal(0, 5, n)
    hrv_sdnn = np.clip(hrv_sdnn, 15.0, 100.0).round(2)

    # Heart rate: inverse of HRV, peaks midday
    hr_base = 72 - 0.5 * hrv_rmssd_day + np.random.normal(0, 5, n)
    heart_rate = np.clip(hr_base, 45.0, 130.0).round(1)

    # Sleep stages (per hour)
    sleep_stages = np.array([
        _sample_sleep_stage(h, 1)[0] for h in hours_of_day
    ])

    # Sleep fragmentation index: higher during awake hours
    sfrag_base = np.where(np.isin(hours_of_day, list(_AWAKE_HOURS)), 0.65, 0.20)
    sleep_fragmentation_index = np.clip(
        sfrag_base + np.random.normal(0, 0.07, n), 0.0, 1.0
    ).round(3)

    # Active energy: zero at night, peaks midday
    activity_pattern = np.clip(
        np.sin(np.pi * (hours_of_day - 6) / 14) * 300, 0, 500
    )
    active_energy_kcal = np.clip(
        activity_pattern + np.random.normal(0, 40, n), 0.0, 500.0
    ).round(1)

    # Step frequency: zero at night
    step_freq = np.clip(
        (active_energy_kcal / 500.0) * 180 + np.random.normal(0, 10, n), 0.0, 200.0
    ).round(1)

    # SpO2 %: generally high, mild nocturnal dip
    spo2_base = 97.5 - 0.5 * (hours_of_day < 6).astype(float)
    spo2_pct = np.clip(spo2_base + np.random.normal(0, 0.6, n), 88.0, 100.0).round(1)

    # Skin temperature: slight rise post-midnight, lower midday
    skin_temp = 36.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, HOURS_PER_DAY))
    skin_temp_celsius = np.tile(skin_temp, N_DAYS) + np.random.normal(0, 0.2, n)
    skin_temp_celsius = np.clip(skin_temp_celsius, 35.0, 38.5).round(2)

    # ── Anomaly injection (last 24 h = rows 696-719) ──────────────────────────
    event_flag = np.zeros(n, dtype=int)
    if is_high_risk:
        anomaly_idx = np.arange(n - HOURS_PER_DAY, n)
        event_flag[anomaly_idx] = 1

        # Degrading HRV – exponential decay toward floor
        decay = np.linspace(1.0, 0.35, HOURS_PER_DAY)
        hrv_rmssd[anomaly_idx] = np.clip(hrv_rmssd[anomaly_idx] * decay, 10.0, 80.0).round(2)
        hrv_sdnn[anomaly_idx] = np.clip(hrv_sdnn[anomaly_idx] * decay, 15.0, 100.0).round(2)

        # Rising heart rate
        hr_rise = np.linspace(0, 22, HOURS_PER_DAY)
        heart_rate[anomaly_idx] = np.clip(
            heart_rate[anomaly_idx] + hr_rise, 45.0, 130.0
        ).round(1)

        # Fragmented sleep throughout anomaly window
        sleep_stages[anomaly_idx] = np.array([
            _sample_sleep_stage(h, 1, fragmented=True)[0]
            for h in hours_of_day[anomaly_idx]
        ])
        sleep_fragmentation_index[anomaly_idx] = np.clip(
            sleep_fragmentation_index[anomaly_idx] + 0.30, 0.0, 1.0
        ).round(3)

        # Mild SpO2 dip
        spo2_pct[anomaly_idx] = np.clip(
            spo2_pct[anomaly_idx] - np.random.uniform(1.5, 4.0, HOURS_PER_DAY),
            88.0, 100.0
        ).round(1)

        # Slight skin temp elevation
        skin_temp_celsius[anomaly_idx] = np.clip(
            skin_temp_celsius[anomaly_idx] + np.random.uniform(0.3, 0.9, HOURS_PER_DAY),
            35.0, 38.5
        ).round(2)

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "patient_id":                 patient_id,
        "timestamp":                  timestamps,
        "hrv_rmssd":                  hrv_rmssd,
        "hrv_sdnn":                   hrv_sdnn,
        "heart_rate":                 heart_rate,
        "sleep_stage":                sleep_stages,
        "sleep_fragmentation_index":  sleep_fragmentation_index,
        "active_energy_kcal":         active_energy_kcal,
        "step_frequency":             step_freq,
        "spo2_pct":                   spo2_pct,
        "skin_temp_celsius":          skin_temp_celsius,
        "event_flag":                 event_flag,
    })
    return df


def generate_wearable(ehr_path: str = EHR_PATH) -> pd.DataFrame:
    """
    Generate wearable data for all patients listed in the EHR CSV.

    Parameters
    ----------
    ehr_path : str
        Path to the EHR CSV file.

    Returns
    -------
    pd.DataFrame
        Combined wearable DataFrame (~360 000 rows for 500 patients).
    """
    ehr_df = pd.read_csv(ehr_path)
    patient_ids = ehr_df["patient_id"].tolist()
    n_total = len(patient_ids)

    # Deterministically label ~20% as high-risk
    n_high_risk = max(1, int(n_total * ANOMALY_FRACTION))
    high_risk_set = set(
        np.random.choice(patient_ids, size=n_high_risk, replace=False)
    )
    print(f"[Wearable] High-risk patients with anomaly injection: {len(high_risk_set)}")

    chunks = []
    for idx, pid in enumerate(patient_ids, start=1):
        if idx % 100 == 0:
            print(f"  … generating patient {idx}/{n_total}")
        chunk = generate_patient_wearable(
            patient_id=pid,
            is_high_risk=(pid in high_risk_set),
        )
        chunks.append(chunk)

    combined = pd.concat(chunks, ignore_index=True)
    return combined


def save_wearable(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    """Save the wearable DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"[Wearable] Saved {len(df):,} rows x {len(df.columns)} cols "
          f"-> {path}  ({size_mb:.1f} MB)")


def print_summary(df: pd.DataFrame) -> None:
    """Print a brief statistical summary of the wearable dataset."""
    print("\n" + "=" * 60)
    print("  Wearable Dataset Summary")
    print("=" * 60)
    print(f"  Total rows         : {len(df):,}")
    print(f"  Patients           : {df['patient_id'].nunique():,}")
    print(f"  Rows per patient   : {ROWS_PER_PATIENT:,}  ({N_DAYS} days × 24 h)")
    print(f"  Event-flagged rows : {df['event_flag'].sum():,}")
    print(f"  HRV RMSSD (mean)   : {df['hrv_rmssd'].mean():.2f} ms")
    print(f"  Heart rate (mean)  : {df['heart_rate'].mean():.1f} bpm")
    print(f"  SpO2 % (mean)      : {df['spo2_pct'].mean():.2f} %")
    print(f"  Sleep stages dist  :")
    stage_counts = df["sleep_stage"].value_counts()
    for stage, cnt in stage_counts.items():
        print(f"    {stage:<8}: {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    print("Generating synthetic wearable time-series data …")
    wearable_df = generate_wearable()
    save_wearable(wearable_df)
    print_summary(wearable_df)
