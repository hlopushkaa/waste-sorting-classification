"""Export a trained run to ONNX and compare CPU speed of PyTorch and ONNX Runtime.

Usage: python -m src.export --config configs/effb0_finetune.yaml [--seed 6545656]

Writes <run_dir>/model.onnx (dynamic batch size) and <run_dir>/export_benchmark.json.
ONNX Runtime fuses BatchNorm into convolutions and uses optimised CPU kernels, so it is
usually faster than eager PyTorch for single-image inference on a laptop or a server
without a GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .models import build_model


def time_it(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    size = cfg["img_size"]

    model = build_model(cfg["backbone"], 6, cfg["strategy"], cfg["dropout"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()

    onnx_path = run_dir / "model.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=args.opset, dynamo=False)
    print(f"saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(8, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(out - ref).max())
    same_class = bool((out.argmax(1) == ref.argmax(1)).all())
    print(f"max |onnx - torch| = {max_diff:.2e}; same predicted class: {same_class}")

    results = {"threads": torch.get_num_threads(), "max_abs_diff": max_diff, "same_class": same_class, "rows": []}
    for batch in (1, 32):
        xb = torch.randn(batch, 3, size, size)
        xn = xb.numpy()
        with torch.no_grad():
            t_torch = time_it(lambda: model(xb), iters=50 if batch > 1 else 100)
        t_onnx = time_it(lambda: sess.run(None, {"image": xn}), iters=50 if batch > 1 else 100)
        for name, t in (("pytorch_cpu", t_torch), ("onnxruntime_cpu", t_onnx)):
            results["rows"].append({"runtime": name, "batch": batch, "ms_per_batch": t,
                                    "ms_per_image": t / batch, "img_per_sec": batch / t * 1e3})
        print(f"batch {batch:>2}: PyTorch {t_torch / batch:.1f} ms/image, "
              f"ONNX Runtime {t_onnx / batch:.1f} ms/image ({t_torch / t_onnx:.2f}x faster)")
    json.dump(results, open(run_dir / "export_benchmark.json", "w"), indent=2)


if __name__ == "__main__":
    main()
