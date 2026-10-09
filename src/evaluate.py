"""Test-set evaluation + inference benchmark for one trained run.

Usage: python -m src.evaluate --config configs/baseline_effb0_fe.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import CLASSES, make_loaders
from .models import build_model
from .train import predict


@torch.no_grad()
def benchmark(model, device, size=224, batch=1, warmup=20, iters=200):
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {"batch": batch, "ms_per_batch": dt * 1e3, "ms_per_image": dt * 1e3 / batch, "img_per_sec": batch / dt}


def plot_cm(cm, path, title):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(6), CLASSES)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, df = make_loaders(cfg["data_root"], cfg["split_csv"], cfg["batch_size"], cfg["num_workers"], cfg["img_size"])
    model = build_model(cfg["backbone"], 6, cfg["strategy"], cfg["dropout"], cfg.get("weights_file")).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    y, p, prob = predict(model, loaders["test"], device)
    rep = classification_report(y, p, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=range(6))
    plot_cm(cm, run_dir / "confusion_matrix.png", f"{cfg['name']} (test)")

    test_df = df[df.split == "test"].reset_index(drop=True)
    wrong = test_df.assign(pred=[CLASSES[i] for i in p], conf=prob.max(1))[p != y]
    wrong.sort_values("conf", ascending=False).to_csv(run_dir / "misclassified.csv", index=False)

    bench = [benchmark(model, device, cfg["img_size"], b) for b in (1, 32)]
    metrics = {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
               "per_class": {c: rep[c] for c in CLASSES}, "confusion_matrix": cm.tolist(),
               "n_test": int(len(y)), "device": str(device), "torch_threads": torch.get_num_threads(),
               "benchmark": bench}
    json.dump(metrics, open(run_dir / "test_metrics.json", "w"), indent=2)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "n_test")}, indent=2))
    print(pd.DataFrame({c: rep[c] for c in CLASSES}).T.round(3))
    print(cm)
    print(bench)


if __name__ == "__main__":
    main()
