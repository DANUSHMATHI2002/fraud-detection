"""Rebuild dashboard/index.html from the latest training outputs (so the dashboard always matches your run)."""
import json
import webbrowser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build(open_browser: bool = False) -> Path:
    t = pd.read_csv(ROOT / "tableau" / "test_predictions.csv")
    heat = [[int(r.hour), r.channel, round(float(r["mean"]) * 100, 2), int(r["count"])]
            for _, r in t.groupby(["hour", "channel"]).is_fraud.agg(["mean", "sum", "count"]).reset_index().iterrows()]
    cat = t.groupby("category").is_fraud.mean().mul(100).round(2).sort_values(ascending=False)
    data = dict(
        metrics=json.loads((ROOT / "outputs" / "metrics.json").read_text()),
        daily=pd.read_csv(ROOT / "tableau" / "daily_summary.csv").round(1).to_dict("records"),
        curve=pd.read_csv(ROOT / "tableau" / "threshold_curve_test.csv").round(4).to_dict("records"),
        seg=pd.read_csv(ROOT / "tableau" / "segment_performance.csv").round(4).to_dict("records"),
        shap=pd.read_csv(ROOT / "tableau" / "shap_importance.csv").head(10).round(4).to_dict("records"),
        models=pd.read_csv(ROOT / "tableau" / "model_comparison.csv").round(3).to_dict("records"),
        heat=heat, cat=[[k, float(v)] for k, v in cat.items()], n=int(len(t)))
    html = (ROOT / "dashboard" / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = ROOT / "dashboard" / "index.html"
    out.write_text(html)
    print(f"Dashboard written: {out}")
    if open_browser:
        webbrowser.open(out.as_uri())
    return out


if __name__ == "__main__":
    build(open_browser=True)
