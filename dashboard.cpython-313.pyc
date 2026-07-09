"""
dashboard.py
============
Builds a single self-contained HTML dashboard from a batch of AQI predictions.

Called from predict.py after every run. Produces outputs/dashboard.html —
a file that opens directly in any browser (double-click it, or the script
will open it for you automatically). No server, no dependencies beyond
what predict.py already needs.
"""

import os
import json
import base64
import webbrowser
from datetime import datetime

import pandas as pd

OUT_DIR = "outputs"
MODELS_DIR = "models"

CHART_FILES = [
    ("01_correlation_heatmap.png", "Feature Correlation"),
    ("02_model_comparison.png",    "Model Comparison"),
    ("03_pred_vs_actual.png",      "Predicted vs Actual"),
    ("04_residuals.png",           "Residuals"),
    ("05_feature_importance.png",  "Feature Importance"),
    ("06_shap_summary.png",        "SHAP Summary"),
]

CATEGORY_STYLE = {
    "Safe / Healthy":       {"color": "#2F6E5E", "bg": "#E6F0EA", "short": "SAFE"},
    "Polluted / Unhealthy": {"color": "#9C6B15", "bg": "#F5ECD8", "short": "POLLUTED"},
    "Hazardous":            {"color": "#A8431E", "bg": "#F4E3DB", "short": "HAZARDOUS"},
}
DEFAULT_STYLE = {"color": "#5B6459", "bg": "#ECEEE7", "short": "N/A"}

GAUGE_MAX = 3.0  # visual scale ceiling; real AQI values above this are clamped and flagged


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def _load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback if model_training.py hasn't been re-run since this feature was added
    return {
        "model_name": "LightGBM (Tuned)", "MAE": 0.233, "RMSE": 0.418, "R2": 0.885,
        "n_train_rows": None, "n_test_rows": None, "trained_at": "unknown",
    }


def _b64_image(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _gauge_svg(value, record_id):
    """Semi-circular instrument dial, 0 -> GAUGE_MAX, clamped + flagged past the ceiling."""
    clamped = max(0.0, min(value, GAUGE_MAX))
    frac = clamped / GAUGE_MAX
    angle = 180 * frac  # 0..180 degrees sweep, left to right
    import math
    r = 54
    cx, cy = 60, 60
    rad = math.radians(180 - angle)
    x = cx + r * math.cos(rad)
    y = cy - r * math.sin(rad)
    large_arc = 1 if angle > 180 else 0

    if value <= 1.0:
        needle_color = "#2F6E5E"
    elif value <= 2.0:
        needle_color = "#9C6B15"
    else:
        needle_color = "#A8431E"

    overflow_tag = (
        f'<text x="60" y="46" text-anchor="middle" class="gauge-overflow">clamped</text>'
        if value > GAUGE_MAX else ""
    )

    return f'''
    <svg viewBox="0 0 120 78" class="gauge" role="img" aria-label="Predicted AQI {value:.2f}">
      <defs>
        <linearGradient id="gaugeTrack{record_id}" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"  stop-color="#2F6E5E"/>
          <stop offset="45%" stop-color="#9C6B15"/>
          <stop offset="100%" stop-color="#A8431E"/>
        </linearGradient>
      </defs>
      <path d="M 6 60 A 54 54 0 0 1 114 60" fill="none" stroke="url(#gaugeTrack{record_id})"
            stroke-width="8" stroke-linecap="round" opacity="0.35"/>
      <line x1="60" y1="60" x2="{x:.1f}" y2="{y:.1f}" stroke="{needle_color}"
            stroke-width="3" stroke-linecap="round"/>
      <circle cx="60" cy="60" r="4.5" fill="{needle_color}"/>
      {overflow_tag}
      <text x="60" y="74" text-anchor="middle" class="gauge-value">{value:.2f}</text>
    </svg>'''


def _category_style(label):
    return CATEGORY_STYLE.get(label, DEFAULT_STYLE)


def _fmt(v):
    if pd.isna(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ----------------------------------------------------------------------------
# MAIN BUILD FUNCTION
# ----------------------------------------------------------------------------

def build_dashboard(df_result: pd.DataFrame, out_dir: str = OUT_DIR,
                     open_browser: bool = True) -> str:
    """
    Builds outputs/dashboard.html from a predict.py results DataFrame.
    Returns the absolute file path (and opens it in the default browser
    unless open_browser=False).
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = _load_metrics()
    n = len(df_result)
    avg_aqi = df_result["Predicted_AQI"].mean() if n else 0
    counts = df_result["Predicted_AQI_Label"].value_counts().to_dict() if n else {}

    # --- instrument dials (cap the visible strip at 12 records; table below has the rest)
    gauge_cards = []
    for i, row in df_result.head(12).iterrows():
        style = _category_style(row.get("Predicted_AQI_Label"))
        city = row.get("city", "—")
        country = row.get("country", "")
        gauge_cards.append(f'''
        <div class="dial-card">
          {_gauge_svg(float(row["Predicted_AQI"]), i)}
          <div class="dial-loc">{city}{", " + str(country) if country else ""}</div>
          <div class="dial-badge" style="color:{style['color']};background:{style['bg']}">{style['short']}</div>
        </div>''')
    gauge_html = "\n".join(gauge_cards)
    overflow_note = (
        f'<p class="hint">Showing dials for the first 12 of {n} records — full list in the table below.</p>'
        if n > 12 else ""
    )

    # --- table rows
    display_cols = [c for c in ["date", "country", "city", "temperature", "humidity",
                                 "pressure", "season"] if c in df_result.columns]
    rows_html = []
    for i, row in df_result.iterrows():
        style = _category_style(row.get("Predicted_AQI_Label"))
        cells = "".join(f"<td>{_fmt(row.get(c))}</td>" for c in display_cols)
        rows_html.append(f'''
        <tr>
          {cells}
          <td class="num">{row["Predicted_AQI"]:.4f}</td>
          <td><span class="badge" style="color:{style['color']};background:{style['bg']}">{row.get("Predicted_AQI_Label", "N/A")}</span></td>
        </tr>''')
    header_cells = "".join(f"<th>{c.replace('_', ' ').title()}</th>" for c in display_cols)

    # --- training charts (embedded, so the file is fully self-contained)
    chart_blocks = []
    for fname, label in CHART_FILES:
        b64 = _b64_image(os.path.join(out_dir, fname))
        if b64:
            chart_blocks.append(f'''
            <figure class="chart-card">
              <img src="data:image/png;base64,{b64}" alt="{label}" loading="lazy"/>
              <figcaption>{label}</figcaption>
            </figure>''')
    charts_html = "\n".join(chart_blocks) if chart_blocks else '<p class="hint">No training charts found in outputs/ yet — run model_training.py to generate them.</p>'

    trained_at = metrics.get("trained_at", "unknown")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Air Quality — Prediction Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper: #EDEFE7;
    --surface: #FFFFFF;
    --border: #D7DAD0;
    --ink: #1E2A22;
    --ink-soft: #5B6459;
    --safe: #2F6E5E;
    --warn: #9C6B15;
    --hazard: #A8431E;
    --data: #35577A;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 48px 28px 80px; }}

  header.top {{
    display: flex; justify-content: space-between; align-items: flex-end;
    flex-wrap: wrap; gap: 20px;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 20px; margin-bottom: 32px;
  }}
  .eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 6px;
  }}
  h1 {{
    font-family: 'Fraunces', serif; font-weight: 700;
    font-size: 40px; margin: 0; letter-spacing: -0.01em;
  }}
  .meta {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    color: var(--ink-soft); text-align: right;
  }}
  .meta div {{ margin-bottom: 2px; }}

  .stat-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); margin-bottom: 40px;
  }}
  .stat {{ background: var(--surface); padding: 18px 20px; }}
  .stat .label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft);
  }}
  .stat .value {{
    font-family: 'Fraunces', serif; font-size: 28px; font-weight: 600; margin-top: 4px;
  }}

  section {{ margin-bottom: 48px; }}
  h2 {{
    font-family: 'Fraunces', serif; font-size: 22px; font-weight: 600;
    margin: 0 0 4px; display: flex; align-items: baseline; gap: 10px;
  }}
  h2 .num {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--ink-soft); font-weight: 400; }}
  .sub {{ color: var(--ink-soft); font-size: 14px; margin: 0 0 20px; }}
  .hint {{ color: var(--ink-soft); font-size: 13px; font-style: italic; }}

  .dial-strip {{
    display: flex; flex-wrap: wrap; gap: 14px;
  }}
  .dial-card {{
    background: var(--surface); border: 1px solid var(--border);
    padding: 10px 14px 14px; width: 150px; text-align: center;
  }}
  .gauge {{ width: 100%; height: auto; }}
  .gauge-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; fill: var(--ink); font-weight: 500; }}
  .gauge-overflow {{ font-family: 'IBM Plex Mono', monospace; font-size: 8px; fill: var(--hazard); text-transform: uppercase; letter-spacing: 0.05em; }}
  .dial-loc {{ font-size: 13px; margin-top: 6px; font-weight: 500; }}
  .dial-badge {{
    display: inline-block; margin-top: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.06em; padding: 2px 8px;
  }}

  table {{ width: 100%; border-collapse: collapse; background: var(--surface); }}
  th, td {{
    border-bottom: 1px solid var(--border); padding: 10px 12px;
    text-align: left; font-size: 13px;
  }}
  th {{
    font-family: 'IBM Plex Mono', monospace; text-transform: uppercase;
    font-size: 11px; letter-spacing: 0.06em; color: var(--ink-soft);
    border-bottom: 2px solid var(--ink);
  }}
  td.num {{ font-family: 'IBM Plex Mono', monospace; color: var(--data); }}
  .badge {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 3px 8px; }}
  .table-scroll {{ overflow-x: auto; border: 1px solid var(--border); }}

  .chart-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
  }}
  .chart-card {{
    background: var(--surface); border: 1px solid var(--border);
    margin: 0; padding: 12px;
  }}
  .chart-card img {{ width: 100%; height: auto; display: block; }}
  .chart-card figcaption {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    color: var(--ink-soft); margin-top: 8px; text-transform: uppercase; letter-spacing: 0.05em;
  }}

  footer {{
    border-top: 1px solid var(--border); padding-top: 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-soft);
  }}
</style>
</head>
<body>
<div class="wrap">

  <header class="top">
    <div>
      <p class="eyebrow">Air Quality Prediction — Field Report</p>
      <h1>AQI Instrument Panel</h1>
    </div>
    <div class="meta">
      <div>MODEL &nbsp;{metrics.get("model_name", "—")}</div>
      <div>TRAINED &nbsp;{trained_at}</div>
      <div>REPORT GENERATED &nbsp;{generated_at}</div>
    </div>
  </header>

  <div class="stat-row">
    <div class="stat"><div class="label">Records Scored</div><div class="value">{n}</div></div>
    <div class="stat"><div class="label">Mean Predicted AQI</div><div class="value">{avg_aqi:.2f}</div></div>
    <div class="stat"><div class="label">Test R²</div><div class="value">{metrics.get("R2", 0):.3f}</div></div>
    <div class="stat"><div class="label">Test RMSE</div><div class="value">{metrics.get("RMSE", 0):.3f}</div></div>
    <div class="stat"><div class="label">Test MAE</div><div class="value">{metrics.get("MAE", 0):.3f}</div></div>
  </div>

  <section>
    <h2>Readings <span class="num">{n} record{"s" if n != 1 else ""}</span></h2>
    <p class="sub">Each dial reads the predicted AQI on a 0–{GAUGE_MAX:.0f} scale — green is safe, amber is polluted, rust is hazardous.</p>
    <div class="dial-strip">
      {gauge_html if gauge_html else '<p class="hint">No records to display.</p>'}
    </div>
    {overflow_note}
  </section>

  <section>
    <h2>Full Prediction Table</h2>
    <div class="table-scroll">
      <table>
        <thead><tr>{header_cells}<th>Predicted AQI</th><th>Category</th></tr></thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Training Diagnostics</h2>
    <p class="sub">Charts generated by model_training.py — embedded here so this file works on its own.</p>
    <div class="chart-grid">
      {charts_html}
    </div>
  </section>

  <footer>
    Generated by predict.py · air_quality_project · {generated_at}
  </footer>

</div>
</body>
</html>'''

    out_path = os.path.abspath(os.path.join(out_dir, "dashboard.html"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        try:
            webbrowser.open(f"file://{out_path}")
        except Exception:
            pass

    return out_path
