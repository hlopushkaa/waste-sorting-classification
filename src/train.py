"""Train one configuration.  Usage: python -m src.train --config configs/baseline_effb0_fe.yaml"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score

from .data import class_weights, make_loaders
from .models import build_model, count_params, param_groups


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def freeze_bn(model: nn.Module):
    """Keep BatchNorm statistics fixed (used when the backbone is frozen)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device)).softmax(1).cpu()
        ys.append(y); ps.append(out.argmax(1)); probs.append(out)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, df = make_loaders(cfg["data_root"], cfg["split_csv"], cfg["batch_size"], cfg["num_workers"], cfg["img_size"])
    model = build_model(cfg["backbone"], 6, cfg["strategy"], cfg["dropout"], cfg.get("weights_file")).to(device)
    total, trainable = count_params(model)
    print(f"params total={total/1e6:.2f}M trainable={trainable/1e3:.1f}K device={device}")

    w = class_weights(df[df.split == "train"].label).to(device) if cfg["class_weighted_loss"] else None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(param_groups(model, cfg["head_lr"], cfg["backbone_lr"]), weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(loaders["train"])
    total_steps, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        if cfg["strategy"] == "feature_extraction":
            freeze_bn(model)
        run_loss, n, correct = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run_loss += loss.item() * len(y); n += len(y); correct += (out.argmax(1) == y).sum().item()
        yv, pv, _ = predict(model, loaders["val"], device)
        val_f1, val_acc = f1_score(yv, pv, average="macro"), accuracy_score(yv, pv)
        rec = {"epoch": epoch, "train_loss": run_loss / n, "train_acc": correct / n,
               "val_acc": val_acc, "val_macro_f1": val_f1, "sec": time.time() - t0}
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stop at epoch {epoch}")
                break

    json.dump({"config": cfg, "seed": seed, "params_total": total, "params_trainable": trainable,
               "best_val_macro_f1": best_f1, "history": history}, open(run_dir / "train_log.json", "w"), indent=2)
    print(f"best val macro-F1 {best_f1:.4f}; checkpoint {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
