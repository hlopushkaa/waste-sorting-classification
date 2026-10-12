# Waste Sorting Image Classification

6-class waste classifier (cardboard, glass, metal, paper, plastic, trash) on [TrashNet](https://github.com/garythung/trashnet),
comparing two ImageNet-pretrained CNN backbones (EfficientNet-B0, ResNet-50) and two fine-tuning strategies
(feature extraction vs. full fine-tuning).

Team: Svetlana Maltseva, Valeria Kolpakova, Alexey Feofanov, Vadim Gafarov 

## Layout

```
configs/                 one YAML per experiment (baseline: baseline_effb0_fe.yaml)
scripts/prepare_data.py  hash, deduplicate and split the dataset -> data/splits.csv
src/data.py              Dataset, transforms, stratified split, class weights
src/models.py            timm backbone + new head, freezing, LR groups
src/train.py             training loop (AdamW, warm-up + cosine, early stopping on val macro-F1)
src/evaluate.py          test metrics, confusion matrix, misclassified list, latency benchmark
runs/<name>_seed<k>/     checkpoints and results (git-ignored except small json/png)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. data 
mkdir -p data/raw && cd data/raw
curl -L -o ds.zip https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip
unzip -q ds.zip && rm -rf __MACOSX ds.zip && cd ../..
python scripts/prepare_data.py --root data/raw/dataset-resized

# 2. ImageNet weights 
mkdir -p weights
curl -L -o weights/efficientnet_b0_ra-3dd342df.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth

# 3. baseline
python -m src.train    --config configs/baseline_effb0_fe.yaml
python -m src.evaluate --config configs/baseline_effb0_fe.yaml
```

## Status

- [x] Data pipeline (deduplication, stratified 70/15/15 split, augmentation)
- [x] Baseline: EfficientNet-B0, frozen backbone (E-FE) — test accuracy 84.7%, macro-F1 0.824
- [ ] E-FT, R-FE, R-FT configurations, 3 seeds each
- [ ] Grad-CAM failure-case analysis
