"""DANN (Domain-Adversarial Neural Network, Ganin 2016)

backbone + multi-level cls (consistent with UnifiedGas)
+ Gradient Reversal Layer (GRL) on f1
+ 2-layer domain discriminator (128 -> 64 -> 1)
+ adversarial loss with progressive λ_p schedule
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
from itertools import cycle

from ._common import (
    maybe_mixup, cls_loss_with_mix,
    FeatureExtractor, Classifier, make_optim, init_weights,
    make_loaders, evaluate, NUM_CLASSES,
)
from ._common import set_seed, cleanup


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lam)


class DomainDiscriminator(nn.Module):
    """f1 (128) -> 64 -> 1 (binary: source vs target)."""

    def __init__(self, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DANNTrainer:
    """DANN trainer: CE + adversarial domain loss with a progressive λ schedule."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.1,
                 device: str = "cuda", lr: float = 1e-3, weight_decay: float = 0.01,
                 dataset: str = "B"):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset = dataset

    def train(self, src_X: torch.Tensor, src_y: torch.Tensor,
              tgt_X: torch.Tensor, tgt_y: torch.Tensor,
              device, config: dict, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset,
                                       dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        domain_disc = DomainDiscriminator(in_dim=128).to(dev)
        init_weights(extractor)
        init_weights(classifier)
        init_weights(domain_disc)

        epochs = config["epochs"]
        bs = config["batch_size"]
        opt, sch = make_optim([extractor, classifier, domain_disc],
                              lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay),
                              total_epochs=epochs)
        bce = nn.BCEWithLogitsLoss()

        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)

        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            # DANN λ schedule: λ_p = 2/(1+exp(-10p)) - 1, p = ep/epochs
            p = ep / max(epochs - 1, 1)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            extractor.train()
            classifier.train()
            domain_disc.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                # Source classification
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))
                feats_s = extractor(xs)
                o1, o2, fo = classifier(feats_s["f1"], feats_s["f2"])
                cls_l = cls_loss_with_mix(o1, o2, fo, ys, _mix)

                # Adversarial domain loss (source + target)
                feats_t = extractor(xt)
                f1_all = torch.cat([feats_s["f1"], feats_t["f1"]], dim=0)
                d_all = torch.cat([torch.ones(xs.size(0), device=dev),
                                    torch.zeros(xt.size(0), device=dev)], dim=0)
                f1_rev = grad_reverse(f1_all, lam)
                d_logit = domain_disc(f1_rev)
                dom_l = bce(d_logit, d_all)

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
