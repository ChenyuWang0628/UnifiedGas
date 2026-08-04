"""Shared components for the deep UDA baselines.

Every baseline in this package is built on the *same* backbone and classifier
heads as UnifiedGas, so a comparison isolates the adaptation objective rather
than network capacity.  Concretely, each baseline reuses:

* the CNN backbone from :mod:`unifiedgas.models` (2-D for Dataset A, 1-D for
  Dataset B),
* the two-level fully connected trunk (256 -> 128 -> 64),
* two linear classifier heads fused with fixed weights (0.6 / 0.4),
* the multi-level classification loss (0.4 / 0.3 / 0.3),
* AdamW with cosine annealing and the same learning rate and weight decay.

Only the alignment objective differs between baselines.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models import MultiLevelFC, build_backbone

__all__ = [
    "FeatureExtractor", "Classifier", "make_optim", "multi_level_cls_loss",
    "init_weights", "make_loaders", "evaluate", "MKMMDLoss", "cleanup",
    "set_seed", "maybe_mixup", "cls_loss_with_mix", "NUM_CLASSES",
]


def set_seed(seed: int = 42):
    """Seed every RNG that affects training."""
    import os
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.__version__ >= "2.0.0":
        torch.use_deterministic_algorithms(True, warn_only=True)


def cleanup():
    """Release memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class FeatureExtractor(nn.Module):
    """Backbone + multi-level trunk, exposing all three feature levels."""

    def __init__(self, dataset: str = "A", dropout_rate: float = 0.1):
        super().__init__()
        self.dataset = dataset.upper()
        self.cnn = build_backbone(self.dataset)
        self.fc = MultiLevelFC(self.cnn.feature_dim, [128, 64], dropout_rate)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        if self.dataset == "A":
            if x.dim() == 2 and x.size(1) == 128:
                x = x.view(x.size(0), 1, 16, 8)
            elif x.dim() == 3:
                x = x.unsqueeze(1)
        else:
            if x.dim() == 2:
                x = x.view(x.size(0), 1, 1, x.size(1))
            elif x.dim() == 3:
                x = x.unsqueeze(1)
        return x

    def forward(self, x: torch.Tensor) -> dict:
        cnn_f = self.cnn(self._reshape(x))
        f1, f2 = self.fc(cnn_f)
        return {"cnn_f": cnn_f, "f1": f1, "f2": f2}


class Classifier(nn.Module):
    """Two linear heads fused with fixed weights, as in UnifiedGas."""

    def __init__(self, num_classes: int = 6, w1: float = 0.6):
        super().__init__()
        self.classifier1 = nn.Linear(128, num_classes)
        self.classifier2 = nn.Linear(64, num_classes)
        self.w1, self.w2 = w1, 1.0 - w1

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        o1 = self.classifier1(f1)
        o2 = self.classifier2(f2)
        return o1, o2, self.w1 * o1 + self.w2 * o2


def make_optim(modules, lr: float = 1e-3, weight_decay: float = 0.01,
               total_epochs: int = 400):
    params = []
    for m in modules:
        params += list(m.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-6)
    return opt, sch


def multi_level_cls_loss(o1, o2, fo, y) -> torch.Tensor:
    """0.4*CE(o1) + 0.3*CE(o2) + 0.3*CE(fused), identical to UnifiedGas."""
    crit = nn.CrossEntropyLoss()
    return 0.4 * crit(o1, y) + 0.3 * crit(o2, y) + 0.3 * crit(fo, y)


def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def make_loaders(src_X, src_y, tgt_X, tgt_y, batch_size: int = 8):
    """Training loaders drop the last partial batch so BatchNorm never sees n=1."""
    src_ld = DataLoader(TensorDataset(src_X, src_y), batch_size=batch_size,
                        shuffle=True, drop_last=True)
    tgt_ds = TensorDataset(tgt_X, tgt_y)
    tgt_train_ld = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_eval_ld = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False)
    return src_ld, tgt_train_ld, tgt_eval_ld


@torch.no_grad()
def evaluate(extractor, classifier, loader, device):
    extractor.eval()
    classifier.eval()
    preds, gts = [], []
    for xb, yb in loader:
        feats = extractor(xb.to(device))
        _, _, fo = classifier(feats["f1"], feats["f2"])
        preds.append(fo.argmax(1).cpu().numpy())
        gts.append(yb.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return float((preds == gts).mean()), preds, gts


class MKMMDLoss(nn.Module):
    """Multi-kernel MMD with the bandwidths used throughout the paper."""

    def __init__(self, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)):
        super().__init__()
        self.sigmas = sigmas

    def forward(self, src, tgt):
        s = F.normalize(src.view(src.size(0), -1), p=2, dim=1)
        t = F.normalize(tgt.view(tgt.size(0), -1), p=2, dim=1)
        vals = []
        for sigma in self.sigmas:
            g = 1.0 / (2 * sigma ** 2)
            Kxx = torch.exp(-g * torch.cdist(s, s, p=2) ** 2)
            Kyy = torch.exp(-g * torch.cdist(t, t, p=2) ** 2)
            Kxy = torch.exp(-g * torch.cdist(s, t, p=2) ** 2)
            vals.append(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())
        return torch.stack(vals).mean()


def maybe_mixup(xs, ys, config, n_source):
    """Source-gated mixup, applied identically to every method for fairness.

    Enabled only when the labeled source is small (``n_source`` at or below
    ``mixup_src_thresh``) and ``mixup_alpha`` is positive; on the large Dataset A
    batches it is therefore inactive.  Returns ``(xs, None)`` when no mixing was
    performed.
    """
    alpha = config.get("mixup_alpha", 0.0)
    thr = config.get("mixup_src_thresh", 100)
    if alpha <= 0 or n_source > thr:
        return xs, None
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xs.size(0), device=xs.device)
    return lam * xs + (1.0 - lam) * xs[perm], (ys, ys[perm], lam)


def cls_loss_with_mix(o1, o2, fo, ys, mix_info) -> torch.Tensor:
    """Multi-level classification loss, mixup-aware.

    Falls back to :func:`multi_level_cls_loss` when ``mix_info`` is ``None``.
    """
    if mix_info is None:
        return multi_level_cls_loss(o1, o2, fo, ys)
    y_a, y_b, lam = mix_info
    return (lam * multi_level_cls_loss(o1, o2, fo, y_a)
            + (1.0 - lam) * multi_level_cls_loss(o1, o2, fo, y_b))


# Default number of classes; every trainer accepts an explicit override.
NUM_CLASSES = 6
