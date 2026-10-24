"""
generate_ehr.py
---------------
Generates a synthetic Electronic Health Record (EHR) dataset of 500 patients
for the Precision Cardiology Digital Twin hackathon project.

Features include demographics, comorbidities, and baseline cardiology metrics
with realistic inter-feature correlations.

Output: data/raw/ehr_data.csv
"""

import os
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "raw", "ehr_data.csv")

N_PATIENTS = 500


def _beta_skewed(low: float, high: float, a: float, b: float, size: int) -> np.ndarray:
    """Sample from a beta distribution scaled to [low, high]."""
    return low + (high - low) * np.random.beta(a, b, size)


def generate_ehr(n: int = N_PATIENTS) -> pd.DataFrame:
    """
    Generate a synthetic EHR DataFrame for *n* patients.

    Parameters
    ----------
    n : int
        Number of patients to generate (default 500).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per patient and the following columns:
        patient_id, age, sex, bmi, hypertension, hyperlipidemia,
        type2_diabetes, smoking_index, prev_ischemic_episodes, lvef_pct,
        resting_hr, resting_ecg_qt_ms, resting_ecg_pr_ms,
        systolic_bp, diastolic_bp.
    """
    # ── Patient IDs ──────────────────────────────────────────────────────────
    patient_ids = [f"PAT-{i:04d}" for i in range(1, n + 1)]

    # ── Age: skewed older (mean ~63) ─────────────────────────────────────────
    # Beta(3,2) skews the distribution toward higher end of [40,85]
    age = _beta_skewed(40, 85, 3, 2, n).astype(int)

    # ── Sex ──────────────────────────────────────────────────────────────────
    sex = np.random.choice(["M", "F"], size=n, p=[0.52, 0.48])

    # ── BMI: slightly right-skewed (overweight population) ───────────────────
    bmi = np.clip(_beta_skewed(18, 42, 2.5, 2, n), 18.0, 42.0).round(1)

    # ── Age-derived risk scalar (0-1) ────────────────────────────────────────
    # Used to correlate comorbidities with age
    age_risk = (age - 40) / 45.0  # 0 at age 40, 1 at age 85

    # ── Hypertension (~55% prevalence, higher with age & bmi) ────────────────
    bmi_risk = (bmi - 18) / 24.0
    htn_prob = np.clip(0.35 + 0.25 * age_risk + 0.10 * bmi_risk, 0.0, 1.0)
    hypertension = (np.random.rand(n) < htn_prob).astype(int)

    # ── Hyperlipidemia (~50% prevalence, correlated with age & bmi) ──────────
    lip_prob = np.clip(0.30 + 0.20 * age_risk + 0.15 * bmi_risk, 0.0, 1.0)
    hyperlipidemia = (np.random.rand(n) < lip_prob).astype(int)

    # ── Type-2 Diabetes (~35% prevalence) ────────────────────────────────────
    t2d_prob = np.clip(0.20 + 0.15 * age_risk + 0.12 * bmi_risk, 0.0, 1.0)
    type2_diabetes = (np.random.rand(n) < t2d_prob).astype(int)

    # ── Smoking Index (pack-years): higher smoking → higher htn/lipid risk ───
    # Modelled as a mixture: ~35% non-smokers/ex-smokers (low values), rest spread
    smoking_base = np.random.exponential(scale=8.0, size=n)
    smoking_index = np.clip(smoking_base, 0.0, 40.0).round(1)

    # Retroactively nudge htn/lipid for heavy smokers
    heavy_smoker_mask = smoking_index > 20
    hypertension[heavy_smoker_mask] = np.where(
        np.random.rand(heavy_smoker_mask.sum()) < 0.65,
        1, hypertension[heavy_smoker_mask]
    )
    hyperlipidemia[heavy_smoker_mask] = np.where(
        np.random.rand(heavy_smoker_mask.sum()) < 0.60,
        1, hyperlipidemia[heavy_smoker_mask]
    )

    # ── Previous Ischemic Episodes (0-5, higher with age & comorbidities) ────
    risk_burden = (
        0.4 * age_risk +
        0.2 * hypertension +
        0.15 * type2_diabetes +
        0.1 * hyperlipidemia +
        0.15 * (smoking_index / 40.0)
    )
    prev_eps_raw = np.random.poisson(lam=risk_burden * 2.5, size=n)
    prev_ischemic_episodes = np.clip(prev_eps_raw, 0, 5).astype(int)

    # ── LVEF %: lower in older / higher-burden patients ──────────────────────
    # Normal LVEF ~55-70; we skew toward normal but include depressed values
    lvef_noise = np.random.normal(0, 5, n)
    lvef_base = 65 - 20 * risk_burden  # ranges ~45-65 for low-high burden
    lvef_pct = np.clip(lvef_base + lvef_noise, 20.0, 75.0).round(1)

    # ── Resting Heart Rate ────────────────────────────────────────────────────
    # Low LVEF patients tend to have compensatory tachycardia
    lvef_norm = (lvef_pct - 20) / 55.0  # 0=very low, 1=normal
    hr_base = 75 - 15 * lvef_norm + np.random.normal(0, 7, n)
    resting_hr = np.clip(hr_base, 45.0, 100.0).round(1)

    # ── QT Interval (ms): ~350-500, prolonged in some HF/medication patients ─
    qt_base = 410 + 20 * (1 - lvef_norm) + np.random.normal(0, 18, n)
    resting_ecg_qt_ms = np.clip(qt_base, 350.0, 500.0).round(1)

    # ── PR Interval (ms): 120-220 ─────────────────────────────────────────────
    pr_base = 160 + np.random.normal(0, 20, n) + 10 * age_risk
    resting_ecg_pr_ms = np.clip(pr_base, 120.0, 220.0).round(1)

    # ── Systolic BP (mmHg) ────────────────────────────────────────────────────
    sbp_base = 125 + 30 * htn_prob + np.random.normal(0, 12, n)
    systolic_bp = np.clip(sbp_base, 90.0, 180.0).round(1)

    # ── Diastolic BP (mmHg) ───────────────────────────────────────────────────
    dbp_base = 0.55 * systolic_bp + 20 + np.random.normal(0, 6, n)
    diastolic_bp = np.clip(dbp_base, 60.0, 120.0).round(1)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "patient_id":              patient_ids,
        "age":                     age,
        "sex":                     sex,
        "bmi":                     bmi,
        "hypertension":            hypertension,
        "hyperlipidemia":          hyperlipidemia,
        "type2_diabetes":          type2_diabetes,
        "smoking_index":           smoking_index,
        "prev_ischemic_episodes":  prev_ischemic_episodes,
        "lvef_pct":                lvef_pct,
        "resting_hr":              resting_hr,
        "resting_ecg_qt_ms":       resting_ecg_qt_ms,
        "resting_ecg_pr_ms":       resting_ecg_pr_ms,
        "systolic_bp":             systolic_bp,
        "diastolic_bp":            diastolic_bp,
    })

    return df


def save_ehr(df: pd.DataFrame, path: str = OUTPUT_PATH) -> None:
    """Save the EHR DataFrame to a CSV file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[EHR] Saved {len(df):,} rows x {len(df.columns)} cols -> {path}")


# ── Summary helper ────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    """Print a brief statistical summary of the generated EHR data."""
    print("\n" + "=" * 60)
    print("  EHR Dataset Summary")
    print("=" * 60)
    print(f"  Patients           : {len(df):,}")
    print(f"  Age range          : {df['age'].min()}–{df['age'].max()} (mean {df['age'].mean():.1f})")
    print(f"  Sex (M/F)          : {(df['sex']=='M').sum()} / {(df['sex']=='F').sum()}")
    print(f"  Hypertension       : {df['hypertension'].mean()*100:.1f}%")
    print(f"  Hyperlipidemia     : {df['hyperlipidemia'].mean()*100:.1f}%")
    print(f"  Type-2 Diabetes    : {df['type2_diabetes'].mean()*100:.1f}%")
    print(f"  LVEF % (mean±std)  : {df['lvef_pct'].mean():.1f} ± {df['lvef_pct'].std():.1f}")
    print(f"  Systolic BP (mean) : {df['systolic_bp'].mean():.1f} mmHg")
    print("=" * 60)


if __name__ == "__main__":
    print("Generating synthetic EHR data …")
    ehr_df = generate_ehr()
    save_ehr(ehr_df)
    print_summary(ehr_df)
