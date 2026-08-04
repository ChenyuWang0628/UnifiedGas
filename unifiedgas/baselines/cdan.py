"""CDAN+E (Long et al. 2018) — Conditional Domain Adversarial Network with Entropy Conditioning

backbone + multi-level cls + adversarial domain loss with **multilinear conditioning**::

    h(x) = f(x) ⊗ softmax(g(x))   # outer product of feature and prediction
    domain discriminator: h -> 1;
    entropy-weighted (CDAN+E):    w(x) = 1 + exp(-H(p(x)))

Reference: Long, M. et al., "Conditional Adversarial Domain Adaptation," NeurIPS 2018.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle

from ._common import (
    maybe_mixup, cls_loss_with_mix,
    FeatureExtractor, Classifier, make_optim, init_weights,
    make_loaders, evaluate, NUM_CLASSES,
)
from .dann import grad_reverse  # Reuse the GRL
from ._common import set_seed, cleanup


def _multilinear(feat: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """h = feat ⊗ pred  -> (B, feat_dim * num_classes)"""
    B = feat.size(0)
    return (feat.unsqueeze(2) * pred.unsqueeze(1)).view(B, -1)


def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """H(p) = -Σ p log p; returns (B,)."""
    return -(p * torch.log(p + eps)).sum(dim=1)


class CDANDomainDiscriminator(nn.Module):
    """h (feat_dim * num_classes) -> 256 -> 1."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class CDANTrainer:
    """CDAN+E: multilinear conditioning + entropy weighting.

    Uses f1 (128) as the feature input; with num_classes=4, h_dim = 128*4 = 512.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.1,
                 device: str = "cuda", lr: float = 1e-3, weight_decay: float = 0.01,
                 use_entropy_weight: bool = True, dataset: str = "B"):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_entropy_weight = use_entropy_weight
        self.dataset = dataset

    def train(self, src_X, src_y, tgt_X, tgt_y, device, config, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset,
                                       dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        domain_disc = CDANDomainDiscriminator(in_dim=128 * self.num_classes,
                                                hidden=256).to(dev)
        init_weights(extractor)
        init_weights(classifier)
        init_weights(domain_disc)

        epochs = config["epochs"]
        bs = config["batch_size"]
        opt, sch = make_optim([extractor, classifier, domain_disc],
                              lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay),
                              total_epochs=epochs)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)

        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            p = ep / max(epochs - 1, 1)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            extractor.train()
            classifier.train()
            domain_disc.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))
                fs = extractor(xs)
                ft = extractor(xt)
                o1_s, o2_s, fo_s = classifier(fs["f1"], fs["f2"])
                o1_t, o2_t, fo_t = classifier(ft["f1"], ft["f2"])
                cls_l = cls_loss_with_mix(o1_s, o2_s, fo_s, ys, _mix)

                # CDAN: multilinear map of f1 and softmax(fo); fused logits are more stable
                p_s = F.softmax(fo_s, dim=1).detach()
                p_t = F.softmax(fo_t, dim=1).detach()
                h_s = _multilinear(fs["f1"], p_s)
                h_t = _multilinear(ft["f1"], p_t)
                h_all = torch.cat([h_s, h_t], dim=0)
                d_all = torch.cat([torch.ones(xs.size(0), device=dev),
                                    torch.zeros(xt.size(0), device=dev)], dim=0)
                h_rev = grad_reverse(h_all, lam)
                d_logit = domain_disc(h_rev)
                bce_per_sample = bce(d_logit, d_all)

                if self.use_entropy_weight:
                    # CDAN+E: w(x) = 1 + exp(-H(p(x)))
                    H_s = _entropy(p_s)
                    H_t = _entropy(p_t)
                    w_all = torch.cat([1.0 + torch.exp(-H_s), 1.0 + torch.exp(-H_t)], dim=0)
                    w_all = w_all / w_all.sum() * w_all.numel()  # Normalize
                    dom_l = (w_all * bce_per_sample).mean()
                else:
                    dom_l = bce_per_sample.mean()

                loss = cls_l + dom_l
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
            acc, preds, _ = evaluate(extractor, classifier, tgt_eval_ld, dev)
            if acc > best_acc:
                best_acc, best_preds = acc, preds

        del extractor, classifier, domain_disc, opt, sch
        cleanup()
        return {"best_acc": float(best_acc), "preds": best_preds}
