"""Data loading and preprocessing for TrashNet."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_index(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan class folders, hash every image and drop exact duplicates.

    Returns (clean_index, removed). If the same file appears under different
    classes, every copy is removed because its label is ambiguous. Same-class
    duplicates keep the first copy.
    """
    root = Path(root)
    rows = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((root / cls).glob("*.jpg")):
            rows.append({"path": str(p.relative_to(root)), "label": label,
                         "class": cls, "md5": _md5(p)})
    df = pd.DataFrame(rows)

    by_hash = defaultdict(set)
    for h, c in zip(df.md5, df["class"]):
        by_hash[h].add(c)
    conflict = df.md5.map(lambda h: len(by_hash[h]) > 1)
    same_dup = df.duplicated("md5", keep="first") & ~conflict
    removed = df[conflict | same_dup].assign(
        reason=lambda d: d.md5.map(lambda h: "cross-class duplicate" if len(by_hash[h]) > 1 else "same-class duplicate"))
    clean = df[~(conflict | same_dup)].reset_index(drop=True)
    return clean, removed


def make_split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Stratified train/val/test split. Adds a 'split' column."""
    trainval, test_df = train_test_split(df, test_size=test, stratify=df.label, random_state=seed)
    train_df, val_df = train_test_split(trainval, test_size=val / (1 - test),
                                        stratify=trainval.label, random_state=seed)
    out = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")])
    return out.sort_values(["split", "path"]).reset_index(drop=True)


def get_transforms(train: bool, size: int = 224):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TrashNetDataset(Dataset):
    def __init__(self, root: str | Path, index: pd.DataFrame, transform=None):
        self.root = Path(root)
        self.items = list(zip(index.path, index.label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(self.root / path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def class_weights(labels, n_classes: int = len(CLASSES)) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1."""
    counts = torch.bincount(torch.as_tensor(list(labels)), minlength=n_classes).float()
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


def make_loaders(root, split_csv, batch_size=32, num_workers=2, size=224):
    df = pd.read_csv(split_csv)
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df.split == split]
        ds = TrashNetDataset(root, part, get_transforms(split == "train", size))
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loaders, df
