"""
heart_widget.py — Holographic 3D Digital Twin Cardiac Visualizer
================================================================
Embeds the high-tech, holographic crystalline human heart digital twin
with rhythmic physiological pulse synchronized to live patient BPM,
seamless dark-background screen blending (no rectangular box edges),
ambient cardiac aura, interactive anatomical hotspots, and clinical HUD.
"""

import os
import base64
import streamlit as st
import streamlit.components.v1 as components

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
_IMAGE_PATH = os.path.join(_STATIC_DIR, "digital_twin_heart.jpg")


@st.cache_data(ttl=3600, show_spinner=False)
def _get_heart_image_data_url() -> str:
    """Read the holographic digital twin image as base64 Data URL."""
    if os.path.exists(_IMAGE_PATH):
        with open(_IMAGE_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    return ""


def _build_hologram_html(risk_score: float, heart_rate: float) -> str:
    """Build the interactive holographic digital twin HTML component."""
    img_data_url = _get_heart_image_data_url()

    risk_score = max(0.0, min(1.0, float(risk_score)))
    heart_rate = max(30.0, min(200.0, float(heart_rate)))
    cycle_period = 60.0 / heart_rate

    # Dynamic status theme based on risk score
    if risk_score > 0.60:
        theme_class = "risk-critical"
        status_tag = "ACUTE ISCHEMIC ANOMALY · HIGH RISK"
        status_color = "#f87171"
        status_bg = "rgba(239, 68, 68, 0.16)"
        status_border = "rgba(239, 68, 68, 0.50)"
        lad_status = "CRITICAL ISCHEMIC HYPOPERFUSION"
        lad_color = "#ef4444"
        lad_pulse = "critical-beacon"
    elif risk_score > 0.30:
        theme_class = "risk-moderate"
        status_tag = "AUTONOMIC SHIFT · ELEVATED"
        status_color = "#fbbf24"
        status_bg = "rgba(245, 158, 11, 0.15)"
        status_border = "rgba(245, 158, 11, 0.45)"
        lad_status = "SUBCLINICAL COMPENSATORY STRAIN"
        lad_color = "#f59e0b"
        lad_pulse = "warning-beacon"
    else:
        theme_class = "risk-low"
        status_tag = "HOLOGRAPHIC TWIN · HOMEOSTASIS"
        status_color = "#00f0ff"
        status_bg = "rgba(0, 240, 255, 0.12)"
        status_border = "rgba(0, 240, 255, 0.35)"
        lad_status = "CORONARY PERFUSION NORMAL"
        lad_color = "#00f0ff"
        lad_pulse = "normal-beacon"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Holographic Cardiac Digital Twin</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{
      width: 100%;
      height: 100%;
      background: #030712;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      user-select: none;
    }}

    #hologram-viewport {{
      width: 100%;
      height: 100%;
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
      background: radial-gradient(circle at 50% 50%, rgba(0, 102, 255, 0.08) 0%, rgba(3, 7, 18, 0.98) 75%);
      overflow: hidden;
      border: 1px solid rgba(0, 240, 255, 0.22);
      border-radius: 8px;
    }}

    /* Cybernetic Grid & Orbital Rings */
    .cyber-grid {{
      position: absolute;
      inset: 0;
      background-image: 
        linear-gradient(rgba(0, 240, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.035) 1px, transparent 1px);
      background-size: 28px 28px;
      pointer-events: none;
    }}

    .orbit-ring {{
      position: absolute;
      border-radius: 50%;
      border: 1px dashed rgba(0, 240, 255, 0.20);
      pointer-events: none;
      animation: rotateRing 32s linear infinite;
    }}
    .orbit-ring-1 {{
      width: 480px;
      height: 480px;
      transform: rotateX(68deg) rotateY(18deg);
    }}
    .orbit-ring-2 {{
      width: 580px;
      height: 580px;
      transform: rotateX(74deg) rotateY(-22deg);
      border-color: rgba(56, 189, 248, 0.15);
      animation-duration: 48s;
      animation-direction: reverse;
    }}
    
    @keyframes rotateRing {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}

    /* Ambient Cardiac Aura (Behind the heart) */
    .ambient-cardiac-glow {{
      position: absolute;
      width: 380px;
      height: 380px;
      border-radius: 50%;
      pointer-events: none;
      background: radial-gradient(circle, rgba(0, 240, 255, 0.24) 0%, rgba(0, 102, 255, 0.12) 45%, transparent 72%);
      animation: auraBreathe {cycle_period:.3f}s ease-in-out infinite;
      z-index: 2;
      filter: blur(14px);
    }}

    .risk-critical .ambient-cardiac-glow {{
      background: radial-gradient(circle, rgba(239, 68, 68, 0.35) 0%, rgba(220, 38, 38, 0.18) 50%, transparent 75%);
    }}

    .risk-moderate .ambient-cardiac-glow {{
      background: radial-gradient(circle, rgba(245, 158, 11, 0.30) 0%, rgba(217, 119, 6, 0.15) 50%, transparent 75%);
    }}

    @keyframes auraBreathe {{
      0%   {{ transform: scale(0.92); opacity: 0.7; }}
      25%  {{ transform: scale(1.08); opacity: 1.0; }}
      50%  {{ transform: scale(0.96); opacity: 0.8; }}
      75%  {{ transform: scale(1.03); opacity: 0.9; }}
      100% {{ transform: scale(0.92); opacity: 0.7; }}
    }}

    /* Main Holographic Heart Visual Stage */
    .heart-stage {{
      position: relative;
      width: 450px;
      height: 450px;
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 5;
    }}

    /* Screen blend mode completely eliminates black JPEG background */
    .heart-image {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      mix-blend-mode: screen;
      transform-origin: 50% 55%;
      animation: cardiacPulse {cycle_period:.3f}s cubic-bezier(0.4, 0.0, 0.2, 1) infinite;
      filter: contrast(1.15) brightness(1.08);
      pointer-events: none;
    }}

    .risk-critical .heart-image {{
      animation: criticalPulse {cycle_period * 0.9:.3f}s cubic-bezier(0.4, 0.0, 0.2, 1) infinite;
    }}

    @keyframes cardiacPulse {{
      0%   {{ transform: scale(1.0); }}
      18%  {{ transform: scale(1.038) rotate(-0.6deg); }}
      36%  {{ transform: scale(0.968) rotate(0.8deg); }}
      54%  {{ transform: scale(1.010); }}
      72%  {{ transform: scale(0.994); }}
      100% {{ transform: scale(1.0); }}
    }}

    @keyframes criticalPulse {{
      0%   {{ transform: scale(1.0); }}
      20%  {{ transform: scale(1.055) rotate(-1.2deg); }}
      40%  {{ transform: scale(0.950) rotate(1.4deg); }}
      60%  {{ transform: scale(1.018); }}
      100% {{ transform: scale(1.0); }}
    }}

    /* Scanline Sweeper */
    .scanline {{
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent, rgba(0, 240, 255, 0.75), transparent);
      opacity: 0.6;
      animation: scanSweep 4.2s linear infinite;
      pointer-events: none;
      z-index: 8;
    }}
    @keyframes scanSweep {{
      0%   {{ top: 0%; opacity: 0; }}
      15%  {{ opacity: 0.7; }}
      85%  {{ opacity: 0.7; }}
      100% {{ top: 100%; opacity: 0; }}
    }}

    /* Interactive Anatomical Hotspots */
    .hotspot {{
      position: absolute;
      width: 13px;
      height: 13px;
      border-radius: 50%;
      background: rgba(0, 240, 255, 0.9);
      border: 2px solid #ffffff;
      box-shadow: 0 0 12px #00f0ff;
      cursor: pointer;
      z-index: 15;
      transition: all 0.2s ease;
    }}
    .hotspot::after {{
      content: '';
      position: absolute;
      inset: -5px;
      border-radius: 50%;
      border: 1px solid rgba(0, 240, 255, 0.6);
      animation: hotspotPing 2s infinite;
    }}
    @keyframes hotspotPing {{
      0% {{ transform: scale(0.8); opacity: 1; }}
      100% {{ transform: scale(2.2); opacity: 0; }}
    }}

    .hotspot.critical-beacon {{
      background: #ef4444;
      border-color: #fee2e2;
      box-shadow: 0 0 16px #ef4444, 0 0 30px rgba(239, 68, 68, 0.6);
    }}
    .hotspot.critical-beacon::after {{
      border-color: rgba(239, 68, 68, 0.8);
    }}

    .hotspot.warning-beacon {{
      background: #f59e0b;
      border-color: #fef3c7;
      box-shadow: 0 0 14px #f59e0b;
    }}
    .hotspot.warning-beacon::after {{
      border-color: rgba(245, 158, 11, 0.7);
    }}

    /* Precise Anatomical Hotspot Coordinates on the image */
    #spot-aorta {{ top: 18%; left: 52%; }}
    #spot-lad   {{ top: 56%; left: 50%; }}
    #spot-lv    {{ top: 74%; left: 66%; }}
    #spot-pa    {{ top: 32%; left: 66%; }}

    /* Tooltip Callouts */
    .hotspot-tooltip {{
      position: absolute;
      background: rgba(8, 13, 26, 0.94);
      border: 1px solid rgba(0, 240, 255, 0.4);
      color: #f1f5f9;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 11px;
      pointer-events: none;
      white-space: nowrap;
      opacity: 0;
      transform: translateY(6px);
      transition: all 0.2s ease;
      z-index: 25;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
    }}
    .hotspot:hover + .hotspot-tooltip {{
      opacity: 1;
      transform: translateY(0);
    }}

    /* Clinical Navigation Toolbar (Top) */
    #top-toolbar {{
      position: absolute;
      top: 10px;
      left: 12px;
      right: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 30;
      pointer-events: none;
    }}
    .nav-group {{
      display: flex;
      gap: 4px;
      background: rgba(8, 13, 26, 0.88);
      backdrop-filter: blur(10px);
      padding: 3px 5px;
      border-radius: 6px;
      border: 1px solid rgba(0, 240, 255, 0.2);
      pointer-events: auto;
    }}
    .nav-btn {{
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 10px;
      font-weight: 700;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: 0.5px;
      transition: all 0.15s ease;
    }}
    .nav-btn:hover {{
      color: #00f0ff;
      background: rgba(0, 240, 255, 0.08);
    }}
    .nav-btn.active {{
      color: #00f0ff;
      background: rgba(0, 240, 255, 0.16);
      border: 1px solid rgba(0, 240, 255, 0.35);
    }}

    .status-badge {{
      background: {status_bg};
      border: 1px solid {status_border};
      color: {status_color};
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.6px;
      padding: 4px 10px;
      border-radius: 6px;
      backdrop-filter: blur(8px);
      pointer-events: auto;
      font-family: 'JetBrains Mono', monospace;
      box-shadow: 0 0 12px rgba(0,0,0,0.4);
    }}

    /* Bottom HUD Telemetry Overlay */
    #bottom-hud {{
      position: absolute;
      bottom: 10px;
      left: 12px;
      right: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 30;
      pointer-events: none;
    }}
    .hud-metrics {{
      display: flex;
      gap: 6px;
      pointer-events: auto;
    }}
    .hud-card {{
      background: rgba(8, 13, 26, 0.88);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(0, 240, 255, 0.18);
      border-radius: 5px;
      padding: 4px 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .hud-label {{
      font-size: 9px;
      font-weight: 600;
      color: #64748b;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      font-family: 'JetBrains Mono', monospace;
    }}
    .hud-val {{
      font-size: 12px;
      font-weight: 800;
      color: #00f0ff;
      font-family: 'JetBrains Mono', monospace;
    }}
    .hud-hint {{
      font-size: 10px;
      color: #64748b;
      letter-spacing: 0.2px;
      font-family: 'JetBrains Mono', monospace;
    }}

    /* Interactive Clinical Inspection Modal */
    #inspect-card {{
      position: absolute;
      bottom: 48px;
      left: 12px;
      right: 12px;
      background: rgba(8, 13, 26, 0.95);
      border: 1px solid rgba(0, 240, 255, 0.35);
      border-radius: 6px;
      padding: 10px 14px;
      display: none;
      z-index: 40;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(12px);
    }}
  </style>
</head>
<body>
  <div id="hologram-viewport" class="{theme_class}">
    <div class="cyber-grid"></div>
    <div class="ambient-cardiac-glow"></div>
    <div class="orbit-ring orbit-ring-1"></div>
    <div class="orbit-ring orbit-ring-2"></div>
    <div class="scanline"></div>

    <!-- Top Toolbar -->
    <div id="top-toolbar">
      <div class="nav-group">
        <button class="nav-btn active" id="btn-core" onclick="setMode('core', this)">Holo Core</button>
        <button class="nav-btn" id="btn-lad" onclick="setMode('lad', this)">LAD Perfusion</button>
        <button class="nav-btn" id="btn-mesh" onclick="setMode('mesh', this)">Mesh Plexus</button>
        <button class="nav-btn" id="btn-pulse" onclick="togglePulse(this)">Pulse: ON</button>
      </div>
      <div class="status-badge">
        ● {status_tag}
      </div>
    </div>

    <!-- Central Heart Digital Twin -->
    <div class="heart-stage" id="stage">
      <img src="{img_data_url}" alt="Holographic Cardiac Digital Twin" class="heart-image" id="heart-img"/>

      <!-- Hotspot 1: Aorta -->
      <div class="hotspot" id="spot-aorta" onclick="showInspect('Aortic Arch & Brachiocephalic Trunk', 'Transvalvular pressure: 120/80 mmHg · Elastin compliance: Normal · Peak velocity: 1.2 m/s', '#00f0ff')"></div>
      <div class="hotspot-tooltip" style="top:10%;left:52%;">
        <strong style="color:#00f0ff;">Aorta & Arch</strong><br>
        <span style="color:#94a3b8;">Ascending Aorta & 3 Branch Arteries</span>
      </div>

      <!-- Hotspot 2: LAD -->
      <div class="hotspot {lad_pulse}" id="spot-lad" onclick="showInspect('Left Anterior Descending (LAD) Artery', 'Status: {lad_status} · Continuous flow index: 0.62 · Dynamic ischemia hazard: {risk_score * 100:.1f}%', '{lad_color}')"></div>
      <div class="hotspot-tooltip" style="top:50%;left:55%;">
        <strong style="color:{lad_color};">LAD Coronary Artery</strong><br>
        <span style="color:#94a3b8;">Anterior Interventricular Septal Trunk</span>
      </div>

      <!-- Hotspot 3: Left Ventricle -->
      <div class="hotspot" id="spot-lv" onclick="showInspect('Left Ventricular Apex', 'Myocardial wall thickness: 11mm · Dynamic torsion: 11.5° · LVEF correlation active', '#00f0ff')"></div>
      <div class="hotspot-tooltip" style="top:74%;left:70%;">
        <strong style="color:#00f0ff;">Left Ventricle</strong><br>
        <span style="color:#94a3b8;">Conical Muscular Chamber & Apex</span>
      </div>

      <!-- Hotspot 4: Pulmonary Trunk -->
      <div class="hotspot" id="spot-pa" onclick="showInspect('Pulmonary Arterial Trunk', 'RV outflow tract conus arteriosus · Deoxygenated return to pulmonary bed', '#00f0ff')"></div>
      <div class="hotspot-tooltip" style="top:28%;left:70%;">
        <strong style="color:#00f0ff;">Pulmonary Trunk</strong><br>
        <span style="color:#94a3b8;">RV Deoxygenated Outflow</span>
      </div>
    </div>

    <!-- Interactive Clinical Inspection Card -->
    <div id="inspect-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span id="inspect-title" style="font-size:11px;font-weight:800;color:#00f0ff;font-family:'JetBrains Mono',monospace;">ANATOMICAL TELEMETRY</span>
        <button onclick="closeInspect()" style="background:transparent;border:none;color:#64748b;font-size:14px;cursor:pointer;">✕</button>
      </div>
      <div id="inspect-body" style="font-size:11px;color:#cbd5e1;margin-top:4px;font-family:'JetBrains Mono',monospace;"></div>
    </div>

    <!-- Bottom HUD -->
    <div id="bottom-hud">
      <div class="hud-metrics">
        <div class="hud-card">
          <span class="hud-label">BPM</span>
          <span class="hud-val">{heart_rate:.0f}</span>
        </div>
        <div class="hud-card">
          <span class="hud-label">LAD THREAT</span>
          <span class="hud-val" style="color:{status_color};">{risk_score * 100:.1f}%</span>
        </div>
        <div class="hud-card">
          <span class="hud-label">CYCLE</span>
          <span class="hud-val" id="cycle-state" style="color:#38bdf8;">SYSTOLE</span>
        </div>
      </div>
      <div class="hud-hint">
        Click nodes for anatomical telemetry
      </div>
    </div>
  </div>

  <script>
    let isPulsing = true;
    const heartImg = document.getElementById('heart-img');
    const cycleState = document.getElementById('cycle-state');
    const inspectCard = document.getElementById('inspect-card');
    const inspectTitle = document.getElementById('inspect-title');
    const inspectBody = document.getElementById('inspect-body');

    function setMode(mode, btn) {{
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');

      if (mode === 'lad') {{
        heartImg.style.filter = 'contrast(1.35) brightness(1.25) drop-shadow(0 0 15px #00f0ff)';
      }} else if (mode === 'mesh') {{
        heartImg.style.filter = 'contrast(1.5) saturate(1.3) brightness(1.15)';
      }} else {{
        heartImg.style.filter = 'contrast(1.15) brightness(1.08)';
      }}
    }}

    function togglePulse(btn) {{
      isPulsing = !isPulsing;
      if (isPulsing) {{
        heartImg.style.animationPlayState = 'running';
        btn.textContent = 'Pulse: ON';
        btn.style.color = '#00f0ff';
      }} else {{
        heartImg.style.animationPlayState = 'paused';
        btn.textContent = 'Pulse: PAUSED';
        btn.style.color = '#94a3b8';
      }}
    }}

    setInterval(() => {{
      if (cycleState && isPulsing) {{
        cycleState.textContent = cycleState.textContent === 'SYSTOLE' ? 'DIASTOLE' : 'SYSTOLE';
      }}
    }}, {cycle_period * 500:.0f});

    function showInspect(title, body, col) {{
      inspectTitle.textContent = title;
      inspectTitle.style.color = col || '#00f0ff';
      inspectBody.textContent = body;
      inspectCard.style.display = 'block';
    }}

    function closeInspect() {{
      inspectCard.style.display = 'none';
    }}
  </script>
</body>
</html>"""


def render_heart_3d(risk_score: float, heart_rate: float = 72.0, height: int = 500):
    """
    Renders the realistic holographic 3D human heart digital twin in a Streamlit HTML component.
    """
    html_content = _build_hologram_html(risk_score=risk_score, heart_rate=heart_rate)
    components.html(html_content, height=height, scrolling=False)
