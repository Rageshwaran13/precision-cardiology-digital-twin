"""
ehr_panel.py — Holographic Clinical Command Panel Component
===========================================================
Renders the patient's baseline EHR profile (LVEF, Blood Pressure with MAP,
ECG intervals, comorbidities, and cardiac anamnesis) styled with
the glowing cyan/cobalt digital twin HUD aesthetic.
"""

from __future__ import annotations
import streamlit as st


def render_ehr_panel(patient_data: dict):
    """
    Renders the clinical command panel as a high-tech holographic HUD.
    All HTML is compiled into a single string with zero leading whitespace to
    prevent Markdown code block interpretation.
    """
    pid        = patient_data.get("patient_id", "PAT-0001")
    age        = patient_data.get("age", 65)
    sex        = patient_data.get("sex", "Male")
    bmi        = float(patient_data.get("bmi", 26.5))
    lvef       = float(patient_data.get("lvef_pct", 55.0))
    resting_hr = float(patient_data.get("resting_hr", 72.0))
    sbp        = float(patient_data.get("sbp", patient_data.get("systolic_bp", 128.0)))
    dbp        = float(patient_data.get("dbp", patient_data.get("diastolic_bp", 82.0)))
    qt_ms      = float(patient_data.get("qt_interval_ms", patient_data.get("resting_ecg_qt_ms", 412.0)))
    pr_ms      = float(patient_data.get("pr_interval_ms", patient_data.get("resting_ecg_pr_ms", 158.0)))
    prev_isch  = int(patient_data.get("prev_ischemic_episodes", 0))
    smoking    = float(patient_data.get("smoking_index", 0.0))

    map_val = dbp + (sbp - dbp) / 3.0

    if lvef >= 55.0:
        lvef_class = "Preserved EF (Normal)"
        lvef_col = "#00f0ff"
    elif lvef >= 45.0:
        lvef_class = "Mildly Reduced EF"
        lvef_col = "#fbbf24"
    else:
        lvef_class = "Severely Impaired EF"
        lvef_col = "#f87171"

    qt_threshold = 460 if sex == "Female" else 450
    qt_prolonged = qt_ms > qt_threshold

    comorbidities = [
        ("Hypertension", bool(patient_data.get("hypertension", 0))),
        ("Type-2 Diabetes", bool(patient_data.get("type2_diabetes", patient_data.get("diabetes", 0)))),
        ("Hyperlipidemia", bool(patient_data.get("hyperlipidemia", 0))),
        ("Prior Ischemia", prev_isch > 0),
        ("Smoking History", smoking > 5.0),
        ("LVEF < 50%", lvef < 50.0),
    ]

    comorb_badges = []
    for name, active in comorbidities:
        if active:
            bg_c = "rgba(239, 68, 68, 0.14)"
            border_c = "rgba(239, 68, 68, 0.40)"
            text_c = "#f87171"
            dot = "●"
        else:
            bg_c = "rgba(0, 240, 255, 0.04)"
            border_c = "rgba(0, 240, 255, 0.15)"
            text_c = "#64748b"
            dot = "○"

        comorb_badges.append(
            f'<span style="background:{bg_c};border:1px solid {border_c};color:{text_c};padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;display:inline-flex;align-items:center;gap:4px;font-family:\'JetBrains Mono\',monospace;">'
            f'<span style="font-size:8px;">{dot}</span> {name}'
            f'</span>'
        )
    comorb_html = "".join(comorb_badges)

    html_lines = [
        '<div style="background:rgba(8,13,26,0.90);backdrop-filter:blur(14px);border:1px solid rgba(0,240,255,0.22);border-radius:8px;padding:14px 16px;box-shadow:0 4px 20px rgba(0,0,0,0.5);margin-bottom:8px;">',
        '  <!-- Patient Summary Ribbon -->',
        '  <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:10px;border-bottom:1px solid rgba(0,240,255,0.12);margin-bottom:12px;">',
        '    <div>',
        f'      <div style="font-size:15px;font-weight:800;color:#f8fafc;font-family:\'JetBrains Mono\',monospace;letter-spacing:0.5px;"><span style="color:#00f0ff;">ID:</span> {pid}</div>',
        f'      <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{age} yrs &nbsp;·&nbsp; {sex} &nbsp;·&nbsp; BMI {bmi:.1f} kg/m²</div>',
        '    </div>',
        '    <div style="text-align:right;">',
        '      <span style="font-size:9px;font-weight:700;letter-spacing:0.8px;color:#00f0ff;background:rgba(0,240,255,0.08);padding:3px 8px;border-radius:4px;border:1px solid rgba(0,240,255,0.25);font-family:\'JetBrains Mono\',monospace;">● EHR SYNCHRONIZED</span>',
        '    </div>',
        '  </div>',
        '  <!-- Section 1: Hemodynamics & Pump Performance -->',
        '  <div style="font-size:10px;font-weight:700;letter-spacing:0.8px;color:#38bdf8;text-transform:uppercase;margin-bottom:8px;font-family:\'JetBrains Mono\',monospace;">CARDIAC PUMP & HEMODYNAMICS</div>',
        '  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="font-size:9px;color:#94a3b8;font-weight:600;">LVEF (ECHO)</div>',
        f'      <div style="font-size:17px;font-weight:800;color:{lvef_col};font-family:\'JetBrains Mono\',monospace;margin:2px 0;">{lvef:.0f}%</div>',
        f'      <div style="font-size:9px;color:#64748b;">{lvef_class}</div>',
        '    </div>',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="font-size:9px;color:#94a3b8;font-weight:600;">BLOOD PRESSURE</div>',
        f'      <div style="font-size:17px;font-weight:800;color:#f1f5f9;font-family:\'JetBrains Mono\',monospace;margin:2px 0;">{sbp:.0f}/{dbp:.0f}</div>',
        f'      <div style="font-size:9px;color:#64748b;">MAP: {map_val:.0f} mmHg</div>',
        '    </div>',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="font-size:9px;color:#94a3b8;font-weight:600;">RESTING HR</div>',
        f'      <div style="font-size:17px;font-weight:800;color:#00f0ff;font-family:\'JetBrains Mono\',monospace;margin:2px 0;">{resting_hr:.0f} <span style="font-size:10px;font-weight:600;color:#64748b;">bpm</span></div>',
        '      <div style="font-size:9px;color:#38bdf8;">Sinus Rhythm</div>',
        '    </div>',
        '  </div>',
        '  <!-- Section 2: Electrophysiology Baseline -->',
        '  <div style="font-size:10px;font-weight:700;letter-spacing:0.8px;color:#38bdf8;text-transform:uppercase;margin-bottom:8px;font-family:\'JetBrains Mono\',monospace;">12-LEAD ELECTROPHYSIOLOGY BASELINE</div>',
        '  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="display:flex;justify-content:space-between;align-items:center;">',
        '        <span style="font-size:9px;color:#94a3b8;font-weight:600;">QTc INTERVAL</span>',
        f'        <span style="font-size:9px;color:{"#f87171" if qt_prolonged else "#00f0ff"};font-family:\'JetBrains Mono\',monospace;">{"PROLONGED" if qt_prolonged else "NORMAL"}</span>',
        '      </div>',
        f'      <div style="font-size:16px;font-weight:800;color:{"#f87171" if qt_prolonged else "#f1f5f9"};font-family:\'JetBrains Mono\',monospace;margin-top:2px;">{qt_ms:.0f} ms</div>',
        f'      <div style="font-size:9px;color:#64748b;">Bazett Ref: &lt;{qt_threshold}ms</div>',
        '    </div>',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="display:flex;justify-content:space-between;align-items:center;">',
        '        <span style="font-size:9px;color:#94a3b8;font-weight:600;">PR INTERVAL</span>',
        '        <span style="font-size:9px;color:#00f0ff;font-family:\'JetBrains Mono\',monospace;">SYNCHRONOUS</span>',
        '      </div>',
        f'      <div style="font-size:16px;font-weight:800;color:#f1f5f9;font-family:\'JetBrains Mono\',monospace;margin-top:2px;">{pr_ms:.0f} ms</div>',
        '      <div style="font-size:9px;color:#64748b;">Ref: 120 - 200 ms</div>',
        '    </div>',
        '  </div>',
        '  <!-- Section 3: Clinical Comorbidities -->',
        '  <div style="font-size:10px;font-weight:700;letter-spacing:0.8px;color:#38bdf8;text-transform:uppercase;margin-bottom:8px;font-family:\'JetBrains Mono\',monospace;">COMORBIDITY MATRIX</div>',
        f'  <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px;">{comorb_html}</div>',
        '  <!-- Section 4: Cardiovascular Anamnesis -->',
        '  <div style="font-size:10px;font-weight:700;letter-spacing:0.8px;color:#38bdf8;text-transform:uppercase;margin-bottom:8px;font-family:\'JetBrains Mono\',monospace;">ANAMNESIS & RISK HISTORY</div>',
        '  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="font-size:9px;color:#94a3b8;">PRIOR ISCHEMIC EPISODES</div>',
        f'      <div style="font-size:15px;font-weight:800;color:{"#f87171" if prev_isch > 0 else "#00f0ff"};font-family:\'JetBrains Mono\',monospace;margin-top:2px;">{prev_isch} <span style="font-size:10px;font-weight:600;color:#64748b;">confirmed</span></div>',
        '    </div>',
        '    <div style="background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.14);border-radius:5px;padding:7px 9px;">',
        '      <div style="font-size:9px;color:#94a3b8;">TOBACCO EXPOSURE</div>',
        f'      <div style="font-size:15px;font-weight:800;color:{"#fbbf24" if smoking > 5 else "#00f0ff"};font-family:\'JetBrains Mono\',monospace;margin-top:2px;">{smoking:.1f} <span style="font-size:10px;font-weight:600;color:#64748b;">pack-years</span></div>',
        '    </div>',
        '  </div>',
        '</div>',
    ]

    clean_html = "\n".join(line.strip() for line in html_lines if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)
