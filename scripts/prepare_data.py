"""Hash, deduplicate and split TrashNet. Writes data/splits.csv and data/removed_duplicates.csv.

Usage: python scripts/prepare_data.py --root data/raw/dataset-resized
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASSES, build_index, make_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/dataset-resized")
ap.add_argument("--out", default="data")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

clean, removed = build_index(a.root)
split = make_split(clean, seed=a.seed)
Path(a.out).mkdir(exist_ok=True)
split.to_csv(Path(a.out) / "splits.csv", index=False)
removed.to_csv(Path(a.out) / "removed_duplicates.csv", index=False)
print(f"kept {len(clean)} images, removed {len(removed)}")
print(removed[["path", "reason"]].to_string(index=False))
print(split.pivot_table(index="class", columns="split", values="path", aggfunc="count")
      .reindex(CLASSES)[["train", "val", "test"]].assign(total=lambda d: d.sum(axis=1)).to_string())
