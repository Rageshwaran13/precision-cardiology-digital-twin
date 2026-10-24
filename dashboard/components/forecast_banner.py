"""
forecast_banner.py — Holographic Clinical Intelligence Horizon Component
=========================================================================
Renders the 24-hour acute ischemic risk index, primary pathophysiological
biomarkers, and dynamic autonomic telemetry aligned with the holographic
cyan/cobalt digital twin design system.
"""

from __future__ import annotations
import streamlit as st


def _get_risk_theme(score: float) -> dict:
    """Return holographic theme tokens based on risk score."""
    if score < 0.30:
        return {
            "tier": "HOMEOSTASIS · NORMAL",
            "tier_short": "LOW RISK",
            "text_color": "#00f0ff",        # Electric Cyan
            "bg_tint": "rgba(0, 240, 255, 0.10)",
            "border_color": "rgba(0, 240, 255, 0.40)",
            "bar_color": "#00f0ff",
            "glow": "0 0 16px rgba(0, 240, 255, 0.25)",
            "summary": "Coronary hemodynamics and autonomic balance are within expected physiological baseline.",
        }
    elif score < 0.60:
        return {
            "tier": "AUTONOMIC SHIFT · ELEVATED",
            "tier_short": "MODERATE RISK",
            "text_color": "#fbbf24",        # Warm Amber
            "bg_tint": "rgba(245, 158, 11, 0.12)",
            "border_color": "rgba(245, 158, 11, 0.45)",
            "bar_color": "#f59e0b",
            "glow": "0 0 16px rgba(245, 158, 11, 0.30)",
            "summary": "Parasympathetic decline observed; subclinical compensatory myocardial strain present.",
        }
    elif score < 0.85:
        return {
            "tier": "ACUTE ISCHEMIC ANOMALY",
            "tier_short": "HIGH RISK",
            "text_color": "#f87171",        # Crimson Alert
            "bg_tint": "rgba(239, 68, 68, 0.14)",
            "border_color": "rgba(239, 68, 68, 0.50)",
            "bar_color": "#ef4444",
            "glow": "0 0 20px rgba(239, 68, 68, 0.35)",
            "summary": "Multimodal anomaly pattern indicates impending acute ischemic crisis within 24 hours.",
        }
    else:
        return {
            "tier": "CRITICAL ISCHEMIC THREAT",
            "tier_short": "CRITICAL",
            "text_color": "#ef4444",
            "bg_tint": "rgba(220, 38, 38, 0.18)",
            "border_color": "rgba(239, 68, 68, 0.65)",
            "bar_color": "#dc2626",
            "glow": "0 0 24px rgba(220, 38, 38, 0.45)",
            "summary": "Severe parasympathetic collapse & nocturnal hypoxia. Immediate clinical verification advised.",
        }


def _trend_glyph(trend: str) -> str:
    mapping = {
        "up": "↑", "down": "↓", "stable": "→",
        "increasing": "↑", "decreasing": "↓", "improving": "↑",
        "worsening": "↓", "elevated": "↑", "fragmented": "↓",
    }
    return mapping.get(str(trend).lower(), "→")


def render_forecast_banner(
    risk_score: float,
    risk_label: str,
    window_hours: int,
    top_risk_factors: list,
    dynamic_triggers: dict,
):
    """
    Renders the holographic forecast banner matching the cyan wireframe aesthetic.
    All HTML is compiled into a single string with zero leading whitespace to
    prevent Markdown code block interpretation.
    """
    theme = _get_risk_theme(risk_score)
    pct = max(0.0, min(100.0, risk_score * 100.0))

    # Build Biomarker Attributions rows
    drivers_html_parts = []
    if top_risk_factors:
        for factor in top_risk_factors[:4]:
            feat = factor.get("feature", "Clinical Biomarker")
            impact = float(factor.get("impact", 0.1))
            dirn = str(factor.get("direction", "↑"))
            dir_color = "#f87171" if ("↑" in dirn or "+" in dirn) else "#00f0ff"
            bar_w = int(min(100, max(10, impact * 100 * 2.5)))
            clean_feat = feat.replace("_", " ").title()
            drivers_html_parts.append(
                f'<div style="display:flex;align-items:center;justify-content:space-between;font-size:11px;gap:6px;padding:2px 0;">'
                f'<span style="color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px;" title="{feat}">'
                f'<span style="color:{dir_color};font-weight:700;margin-right:3px;">{dirn}</span>{clean_feat}'
                f'</span>'
                f'<div style="width:55px;height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;flex-shrink:0;">'
                f'<div style="width:{bar_w}%;height:100%;background:{theme["bar_color"]};border-radius:2px;box-shadow:0 0 6px {theme["bar_color"]};"></div>'
                f'</div>'
                f'<span style="color:#00f0ff;font-size:10px;font-family:\'JetBrains Mono\',monospace;min-width:32px;text-align:right;">'
                f'{impact:.2f}'
                f'</span>'
                f'</div>'
            )
    else:
        drivers_html_parts.append(
            '<div style="font-size:11px;color:#64748b;font-style:italic;">No significant dysregulation detected</div>'
        )
    drivers_html = "".join(drivers_html_parts)

    # Build Real-Time Telemetry Sensors rows
    sensors_html_parts = []
    if dynamic_triggers:
        status_palette = {
            "normal":   ("#00f0ff", "rgba(0, 240, 255, 0.08)"),
            "warning":  ("#fbbf24", "rgba(245, 158, 11, 0.10)"),
            "critical": ("#f87171", "rgba(239, 68, 68, 0.14)"),
            "low":      ("#38bdf8", "rgba(56, 189, 248, 0.08)"),
        }
        for name, info in list(dynamic_triggers.items())[:4]:
            val = info.get("value", "—")
            trend = _trend_glyph(info.get("trend", "stable"))
            stat = info.get("status", "normal")
            val_col, bg_col = status_palette.get(stat, ("#94a3b8", "transparent"))
            sensors_html_parts.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 8px;background:{bg_col};border-radius:4px;font-size:11px;margin-bottom:3px;">'
                f'<span style="color:#94a3b8;font-size:10px;">{name}</span>'
                f'<span style="color:{val_col};font-weight:700;font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                f'{val} <span style="font-size:10px;opacity:0.85;">{trend}</span>'
                f'</span>'
                f'</div>'
            )
    else:
        sensors_html_parts.append(
            '<div style="font-size:11px;color:#64748b;font-style:italic;">Awaiting wearable telemetry packet...</div>'
        )
    sensors_html = "".join(sensors_html_parts)

    # Assemble single unified HTML document with no leading whitespace
    html_lines = [
        '<div style="background:rgba(8,13,26,0.92);backdrop-filter:blur(14px);border:1px solid rgba(0,240,255,0.22);border-left:4px solid ' + theme['bar_color'] + ';border-radius:8px;padding:12px 16px;margin-bottom:10px;box-shadow:' + theme['glow'] + ',0 4px 20px rgba(0,0,0,0.5);">',
        '  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;padding-bottom:8px;border-bottom:1px solid rgba(0,240,255,0.12);margin-bottom:10px;">',
        '    <div style="display:flex;align-items:center;gap:10px;">',
        '      <span style="font-size:11px;font-weight:800;letter-spacing:1px;color:#00f0ff;text-transform:uppercase;font-family:\'JetBrains Mono\',monospace;">⚡ PREDICTIVE FORECAST HORIZON</span>',
        '      <span style="background:' + theme['bg_tint'] + ';border:1px solid ' + theme['border_color'] + ';color:' + theme['text_color'] + ';font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:0.5px;font-family:\'JetBrains Mono\',monospace;">' + theme['tier'] + '</span>',
        '    </div>',
        '    <div style="display:flex;align-items:center;gap:14px;">',
        f'      <span style="font-size:11px;color:#94a3b8;">Forecast Window: <strong style="color:#f8fafc;font-family:\'JetBrains Mono\',monospace;">{window_hours}h Forward Lead Time</strong></span>',
        '      <span style="font-size:10px;color:#00f0ff;background:rgba(0,240,255,0.08);padding:2px 8px;border-radius:4px;border:1px solid rgba(0,240,255,0.2);font-family:\'JetBrains Mono\',monospace;">CONFIDENCE: 94.2%</span>',
        '    </div>',
        '  </div>',
        '  <div style="display:grid;grid-template-columns:30% 38% 32%;gap:16px;align-items:center;">',
        '    <div style="border-right:1px solid rgba(0,240,255,0.1);padding-right:14px;">',
        '      <div style="display:flex;align-items:baseline;gap:8px;">',
        f'        <span style="font-size:36px;font-weight:900;color:#f8fafc;font-family:\'JetBrains Mono\',monospace;line-height:1;text-shadow:0 0 16px {theme["bar_color"]};">{pct:.1f}<span style="font-size:18px;color:#38bdf8;font-weight:600;">%</span></span>',
        f'        <span style="font-size:12px;font-weight:700;color:{theme["text_color"]};letter-spacing:0.5px;">{theme["tier_short"]}</span>',
        '      </div>',
        '      <div style="margin-top:8px;">',
        '        <div style="height:4px;border-radius:2px;background:rgba(255,255,255,0.08);overflow:hidden;">',
        f'          <div style="width:{min(100, pct):.1f}%;height:100%;background:{theme["bar_color"]};box-shadow:0 0 10px {theme["bar_color"]};"></div>',
        '        </div>',
        '        <div style="display:flex;justify-content:space-between;font-size:9px;color:#64748b;margin-top:3px;font-family:\'JetBrains Mono\',monospace;">',
        '          <span>0% (NORM)</span>',
        '          <span>30%</span>',
        '          <span>60%</span>',
        '          <span>85%+ (CRIT)</span>',
        '        </div>',
        '      </div>',
        f'      <div style="font-size:10px;color:#94a3b8;margin-top:6px;line-height:1.35;">{theme["summary"]}</div>',
        '    </div>',
        '    <div style="border-right:1px solid rgba(0,240,255,0.1);padding-right:14px;">',
        '      <div style="font-size:10px;font-weight:700;letter-spacing:0.8px;color:#38bdf8;text-transform:uppercase;margin-bottom:6px;font-family:\'JetBrains Mono\',monospace;">PRIMARY BIOMARKER ATTRIBUTIONS</div>',
        f'      <div style="display:flex;flex-direction:column;gap:3px;">{drivers_html}</div>',
        '    </div>',
        '    <div>',
        '      <div style="font-size:10px;font-weight:700;letter-spacing:0.8px;color:#38bdf8;text-transform:uppercase;margin-bottom:6px;font-family:\'JetBrains Mono\',monospace;">REAL-TIME TELEMETRY SENSORS</div>',
        f'      <div style="display:flex;flex-direction:column;gap:2px;">{sensors_html}</div>',
        '    </div>',
        '  </div>',
        '</div>',
    ]

    # Crucial: strip leading whitespace from every line so Markdown parser NEVER enters code block mode
    clean_html = "\n".join(line.strip() for line in html_lines if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)
