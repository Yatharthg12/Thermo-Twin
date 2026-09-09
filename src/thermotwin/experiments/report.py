"""Generate a self-contained offline HTML research run report."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _table(path: Path) -> str:
    if not path.is_file():
        return "<p class='muted'>No table was produced.</p>"
    frame = pd.read_csv(path)
    return frame.to_html(index=False, border=0, classes="data-table", na_rep="Unavailable")


def _image(path: Path, alt: str) -> str:
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"<img src='data:image/png;base64,{encoded}' alt='{html.escape(alt)}'>"


def generate_report(run_dir: str | Path, summary: dict[str, Any]) -> Path:
    """Embed measured tables/figures and caveats so the report opens without a server."""
    root = Path(run_dir)
    sections = [
        ("1 · Physics and numerical verification", "experiment_1_physics.csv", "physics_verification.png"),
        ("2 · Forecast model benchmark", "experiment_2_forecasts.csv", "forecast_benchmark.png"),
        ("2b · Forecast distribution-shift coverage", "experiment_2_by_scenario.csv", ""),
        ("3 · Horizon sensitivity and uncertainty", "experiment_3_horizons.csv", "horizon_error.png"),
        ("3b · Any-window hotspot probability quality", "experiment_3_hotspot.csv", ""),
        ("4 · Closed-loop controller comparison", "experiment_4_control.csv", "control_energy.png"),
        ("5 · Ablation", "experiment_5_ablation.csv", "ablation_energy.png"),
        ("6 · Robustness and topology scale", "experiment_6_robustness.csv", "scale_runtime.png"),
    ]
    body = "".join(f"<section><h2>{title}</h2>{_image(root / image, title)}<div class='table-wrap'>{_table(root / table)}</div></section>" for title, table, image in sections)
    metadata = html.escape(json.dumps(summary.get("provenance", {}), indent=2))
    profile = html.escape(str(summary.get("profile", "unknown")))
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ThermoTwin experiment report</title><style>
body{{margin:0;background:#07111f;color:#dfeafb;font:15px system-ui,sans-serif;line-height:1.55}}main{{max-width:1120px;margin:auto;padding:40px 24px}}
h1{{font-size:2.2rem;margin-bottom:.2rem}}h2{{color:#80d8ff;margin-top:0}}.lede,.muted{{color:#9db0c8}}section{{background:#0d1b2d;border:1px solid #203752;border-radius:14px;padding:22px;margin:20px 0;overflow:hidden}}
img{{display:block;max-width:100%;background:white;border-radius:8px;margin:14px auto}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #29425d;text-align:right;white-space:nowrap}}th{{color:#80d8ff}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#081422;padding:14px;border-radius:8px}}
</style></head><body><main><p class='lede'>THERMOTWIN / PIPELINE EVIDENCE</p><h1>Simulated digital-twin experiment report</h1>
<p>Profile: <strong>{profile}</strong>. These results describe a software-simulated plant and are not physical facility validation. Smoke results verify the pipeline and are not publication-level statistical evidence.</p>
<section><h2>Methods and information boundary</h2><p>The hidden mismatched plant produces truth. Policies receive noisy/missing observations and bounded estimates only. Future demand and realized disturbances are withheld; candidate actions are evaluated in isolated causal rollouts. Temperature is an effective rack/exhaust thermal node; inlet temperature is separately derived from supply and recirculation.</p></section>
{body}<section><h2>Caveats</h2><p>Parameters are illustrative and not empirically calibrated. Humidity is contextual only. PUE is a cooling-only overhead approximation excluding other facility loads. Hard-limit violations are an experiment proxy, not a service-level agreement. Empirical intervals do not provide unconditional guarantees under temporal dependence or distribution shift.</p></section>
<section><h2>Provenance</h2><pre>{metadata}</pre></section></main></body></html>"""
    path = root / "report.html"
    temp = root / "report.html.tmp"
    temp.write_text(document, encoding="utf-8")
    temp.replace(path)
    return path
