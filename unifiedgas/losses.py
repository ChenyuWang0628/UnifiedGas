"""Alignment and regularization losses used by UnifiedGas.

The training objective is

    L(t) = L_cls + lambda_align(t) * (L_mmd + L_coral) + lambda_ctr * L_ctr

where ``L_cls`` is the multi-level classification loss over the two classifier
heads and their fusion, ``L_mmd`` is the hierarchical multi-kernel MMD summed
over three feature levels, ``L_coral`` aligns second-order statistics, and
``L_ctr`` is the dual-level center regularizer (computed inside
``GasClassifier``).  ``lambda_align`` follows the warmup then ramp-up schedule
in :func:`alignment_weight`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "mk_mmd",
    "coral_loss",
    "hierarchical_mk_mmd",
    "multi_level_cls_loss",
    "alignment_weight",
    "DEFAULT_SIGMAS",
]

DEFAULT_SIGMAS = (0.1, 0.5, 1.0, 2.0, 5.0)


def mk_mmd(src: torch.Tensor, tgt: torch.Tensor,
           sigmas=DEFAULT_SIGMAS) -> torch.Tensor:
    """Multi-kernel MMD with fixed Gaussian bandwidths.

    Both feature sets are L2-normalized first, which keeps the kernel
    bandwidths meaningful across the three feature levels (256/128/64-D).
    """
    s = F.normalize(src.view(src.size(0), -1), p=2, dim=1)
    t = F.normalize(tgt.view(tgt.size(0), -1), p=2, dim=1)
    vals = []
    for sigma in sigmas:
        g = 1.0 / (2 * sigma ** 2)
        Kxx = torch.exp(-g * torch.cdist(s, s, p=2) ** 2)
        Kyy = torch.exp(-g * torch.cdist(t, t, p=2) ** 2)
        Kxy = torch.exp(-g * torch.cdist(s, t, p=2) ** 2)
        vals.append(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())
    return torch.stack(vals).mean()


def hierarchical_mk_mmd(src_feats: list[torch.Tensor], tgt_feats: list[torch.Tensor],
                        sigmas=DEFAULT_SIGMAS) -> torch.Tensor:
    """Average MK-MMD over the three feature levels [cnn_f, f1, f2]."""
    return sum(mk_mmd(a, b, sigmas) for a, b in zip(src_feats, tgt_feats)) / len(src_feats)


def coral_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """CORAL: squared Frobenius distance between the two covariance matrices."""
    d = src.size(1)
    src_c = src - src.mean(0, keepdim=True)
    tgt_c = tgt - tgt.mean(0, keepdim=True)
    cov_s = src_c.t() @ src_c / max(src.size(0) - 1, 1)
    cov_t = tgt_c.t() @ tgt_c / max(tgt.size(0) - 1, 1)
    return ((cov_s - cov_t) ** 2).sum() / (4 * d * d)


def multi_level_cls_loss(out1: torch.Tensor, out2: torch.Tensor, fused: torch.Tensor,
                         y: torch.Tensor, weights=(0.4, 0.3, 0.3)) -> torch.Tensor:
    """Weighted cross-entropy over the two heads and the fused logits.

    The default 4:3:3 ratio is normalized to sum to one so that the
    classification term stays on the same scale as the alignment terms.
    """
    crit = nn.CrossEntropyLoss()
    a1, a2, a3 = weights
    total = float(a1 + a2 + a3)
    return (a1 * crit(out1, y) + a2 * crit(out2, y) + a3 * crit(fused, y)) / total * 1.0


def alignment_weight(epoch: int, total_epochs: int, warmup_frac: float = 0.25) -> float:
    """Alignment weight schedule: zero during warmup, then a sigmoid ramp-up.

    Returns 0 for the first ``warmup_frac`` of training so the classifier can
    form class structure before alignment starts pulling the two domains
    together; afterwards it follows the standard 2/(1+exp(-10p)) - 1 ramp.
    """
    if epoch < max(int(total_epochs * warmup_frac), 1):
        return 0.0
    p = epoch / max(total_epochs - 1, 1)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
