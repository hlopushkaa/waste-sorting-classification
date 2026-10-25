"""Classify your own photos with a trained run.

Usage:
  python -m src.predict --config configs/effb0_finetune.yaml photo1.jpg photo2.jpg
  python -m src.predict --config configs/effb0_finetune.yaml my_photos/ --onnx --tta

--onnx uses <run_dir>/model.onnx (create it with src.export) instead of PyTorch.
--tta averages the prediction for the image and its horizontal mirror.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .data import CLASSES, get_transforms
from .models import build_model

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("images", nargs="+", help="image files or folders")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"

    files = collect(args.images)
    if not files:
        raise SystemExit("no images found")
    tf = get_transforms(train=False, size=cfg["img_size"])
    x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files])
    if args.tta:
        x = torch.cat([x, torch.flip(x, dims=[3])])

    t0 = time.perf_counter()
    if args.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(run_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        t0 = time.perf_counter()
        logits = torch.from_numpy(sess.run(None, {"image": x.numpy()})[0])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg["backbone"], 6, cfg["strategy"], cfg["dropout"], pretrained=False)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        model.to(device).eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x.to(device)).cpu()
    elapsed = (time.perf_counter() - t0) * 1e3
    prob = logits.softmax(1)
    if args.tta:
        prob = (prob[:len(files)] + prob[len(files):]) / 2

    for f, pr in zip(files, prob.numpy()):
        top = np.argsort(-pr)[:2]
        print(f"{f}: {CLASSES[top[0]]} ({pr[top[0]]:.2f}), then {CLASSES[top[1]]} ({pr[top[1]]:.2f})")
    print(f"{len(files)} image(s) in {elapsed:.0f} ms ({elapsed / len(files):.1f} ms per image, "
          f"{'ONNX Runtime' if args.onnx else 'PyTorch'}{', TTA' if args.tta else ''})")


if __name__ == "__main__":
    main()
