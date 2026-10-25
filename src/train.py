"""Train one configuration.  Usage: python -m src.train --config configs/baseline_effb0_fe.yaml"""
from __future__ import annotations

import argparse
import contextlib
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


def autocast(device: torch.device, enabled: bool):
    """float16 mixed precision on CUDA; a no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, loader, device, amp=False, channels_last=False, tta=False):
    """Return labels, predictions and softmax probabilities.

    tta=True averages the probabilities of each image and its horizontal mirror.
    """
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast(device, amp):
            out = model(x).float().softmax(1)
            if tta:
                out = (out + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        out = out.cpu()
        ys.append(y); ps.append(out.argmax(1)); probs.append(out)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override the config's epoch budget")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="quick smoke test: stop each epoch after this many training batches")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    channels_last = bool(cfg.get("channels_last", False))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 224x224 input: let cuDNN pick the fastest kernels
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, df = make_loaders(cfg["data_root"], cfg["split_csv"], cfg["batch_size"], cfg["num_workers"], cfg["img_size"],
                               cfg.get("persistent_workers", False))
    model = build_model(cfg["backbone"], 6, cfg["strategy"], cfg["dropout"], cfg.get("weights_file")).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    total, trainable = count_params(model)
    print(f"params total={total/1e6:.2f}M trainable={trainable/1e3:.1f}K device={device} "
          f"amp={use_amp} channels_last={channels_last}")

    w = class_weights(df[df.split == "train"].label).to(device) if cfg["class_weighted_loss"] else None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(param_groups(model, cfg["head_lr"], cfg["backbone_lr"]), weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = len(loaders["train"]) if args.limit_batches is None else min(args.limit_batches, len(loaders["train"]))
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
        for step, (x, y) in enumerate(loaders["train"]):
            if args.limit_batches is not None and step >= args.limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with autocast(device, use_amp):
                out = model(x)
                loss = criterion(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y); n += len(y); correct += (out.argmax(1) == y).sum().item()
        yv, pv, _ = predict(model, loaders["val"], device, use_amp, channels_last)
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
               "device": str(device), "amp": use_amp, "channels_last": channels_last,
               "train_seconds": sum(r["sec"] for r in history),
               "best_val_macro_f1": best_f1, "history": history}, open(run_dir / "train_log.json", "w"), indent=2)
    print(f"best val macro-F1 {best_f1:.4f}; checkpoint {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
