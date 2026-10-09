"""Model factory: pretrained backbone + new 6-class head."""
from __future__ import annotations

import timm
import torch.nn as nn

# timm names for the two planned backbones
BACKBONES = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "resnet50": "resnet50.a1_in1k",
}


def build_model(backbone: str, num_classes: int = 6, strategy: str = "feature_extraction",
                dropout: float = 0.3, weights_file: str | None = None) -> nn.Module:
    """Create an ImageNet-pretrained backbone with a fresh classifier head.

    strategy: "feature_extraction" freezes every backbone parameter;
              "full_finetune" leaves all parameters trainable.
    weights_file: optional local .pth with ImageNet weights (for offline machines).
    """
    kwargs = dict(pretrained=True, num_classes=num_classes, drop_rate=dropout)
    if weights_file:
        kwargs["pretrained_cfg_overlay"] = dict(file=weights_file)
    model = timm.create_model(BACKBONES[backbone], **kwargs)

    head = model.get_classifier()
    # timm's EfficientNet init scales head weights by 1/sqrt(num_classes); with only 6
    # outputs this gives initial logits of magnitude ~5-10 and a starting loss of ~5.
    # A small normal init starts training from near-uniform predictions instead.
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    if strategy == "feature_extraction":
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    elif strategy != "full_finetune":
        raise ValueError(f"unknown strategy {strategy}")
    return model


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    """Discriminative learning rates: head vs. backbone (only trainable params)."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids]
    body = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    groups = [{"params": head, "lr": head_lr}]
    if body:
        groups.append({"params": body, "lr": backbone_lr})
    return groups


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
