"""Print one comparison table for every evaluated run in runs/.

Usage: python scripts/compare_runs.py [--runs runs] [--csv runs/comparison.csv]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--csv", default=None)
a = ap.parse_args()

rows = []
for f in sorted(Path(a.runs).glob("*/test_metrics*.json")):
    m = json.load(open(f))
    log_file = f.parent / "train_log.json"
    log = json.load(open(log_file)) if log_file.exists() else {}
    bench = {(b["batch"], b.get("precision", "fp32")): b["ms_per_image"] for b in m.get("benchmark", [])}
    row = {"run": f.parent.name + (" +TTA" if m.get("tta") else ""),
           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
           "errors": m["n_test"] - round(m["accuracy"] * m["n_test"]),
           "trainable_params": log.get("params_trainable"),
           "train_s": log.get("train_seconds", sum(r["sec"] for r in log.get("history", []))) or None,
           "best_epoch": max(log["history"], key=lambda r: r["val_macro_f1"])["epoch"] if log.get("history") else None,
           "ms_img_b1": bench.get((1, "fp32")), "ms_img_b32": bench.get((32, "fp32")),
           "ms_img_b32_fp16": bench.get((32, "fp16")), "device": m.get("device")}
    row.update({f"f1_{c}": m["per_class"][c]["f1-score"] for c in CLASSES})
    rows.append(row)

if not rows:
    raise SystemExit(f"no test_metrics*.json found under {a.runs}/")
df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(df.round(4).to_string(index=False))
if a.csv:
    df.to_csv(a.csv, index=False)
    print(f"saved {a.csv}")
