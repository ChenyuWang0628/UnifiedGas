"""SDAT (Smooth Domain Adversarial Training, Rangwani et al. ICML 2022).

SDAT applies sharpness-aware minimization (SAM) to the **task loss** within the
DANN adversarial framework. It first applies a rho-normalized weight perturbation
along the classification-loss gradient, then recomputes and backpropagates the
classification loss at the perturbed point to favor flat minima. The domain
adversarial term itself is not smoothed, which distinguishes SDAT from directly
applying SAM to the full objective.

Reference: Rangwani, H. et al., "A Closer Look at Smoothness in Domain
Adversarial Training," ICML 2022. Uses the shared backbone, multi-level
classifier, and best-selection protocol for a backbone-controlled comparison.
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
from .dann import DomainDiscriminator, grad_reverse
from ._common import set_seed, cleanup


@torch.no_grad()
def _sam_ascent(params, rho: float):
    """Take a rho-normalized ascent step along the current gradient and return the perturbations."""
    grad_norm = torch.norm(
        torch.stack([p.grad.norm(p=2) for p in params if p.grad is not None]), p=2
    )
    scale = rho / (grad_norm + 1e-12)
    eps_list = []
    for p in params:
        if p.grad is None:
            eps_list.append(None)
            continue
        e = p.grad * scale
        p.add_(e)
        eps_list.append(e)
    return eps_list


@torch.no_grad()
def _sam_descent(params, eps_list):
    """Revert the perturbations applied by _sam_ascent."""
    for p, e in zip(params, eps_list):
        if e is not None:
            p.sub_(e)


class SDATTrainer:
    """SDAT: DANN adversarial training + SAM smoothing of the task loss."""

    def __init__(self, num_classes: int = NUM_CLASSES, dataset: str = "A",
                 dropout_rate: float = 0.1, lr: float = 1e-3, weight_decay: float = 0.01,
                 rho: float = 0.02):
        self.num_classes = num_classes
        self.dataset = dataset
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.rho = rho

    def train(self, src_X, src_y, tgt_X, tgt_y, device, config: dict, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset, dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        discriminator = DomainDiscriminator().to(dev)
        init_weights(extractor)
        init_weights(classifier)
        init_weights(discriminator)

        epochs = config["epochs"]
        bs = config["batch_size"]
        rho = config.get("sdat_rho", self.rho)
        opt, sch = make_optim([extractor, classifier, discriminator],
                              lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay),
                              total_epochs=epochs)
        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)
        bce = nn.BCEWithLogitsLoss()

        # Apply SAM only to the feature extractor and classifier parameters used by the task loss
        task_params = [p for m in (extractor, classifier) for p in m.parameters()]

        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            p = ep / max(epochs - 1, 1)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0   # Same λ schedule as the other baselines
            extractor.train()
            classifier.train()
            discriminator.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))

                # ---- Step 1: task-loss gradients for the SAM ascent step ----
                opt.zero_grad()
                fs = extractor(xs)
                o1, o2, fo = classifier(fs["f1"], fs["f2"])
                cls_l = cls_loss_with_mix(o1, o2, fo, ys, _mix)
                cls_l.backward()
                eps_list = _sam_ascent(task_params, rho)

                # ---- Step 2: recompute the task and domain-adversarial losses at the perturbed point ----
                opt.zero_grad()
                fs = extractor(xs)
                o1, o2, fo = classifier(fs["f1"], fs["f2"])
                cls_l2 = cls_loss_with_mix(o1, o2, fo, ys, _mix)
                ft = extractor(xt)
                d_s = discriminator(grad_reverse(fs["f1"], lam))
                d_t = discriminator(grad_reverse(ft["f1"], lam))
                dom_l = 0.5 * (bce(d_s, torch.ones_like(d_s)) +
                               bce(d_t, torch.zeros_like(d_t)))
                (cls_l2 + dom_l).backward()

                # Apply gradients computed at the perturbed point to the original weights
                _sam_descent(task_params, eps_list)
                opt.step()
            sch.step()
            acc, preds, _ = evaluate(extractor, classifier, tgt_eval_ld, dev)
            if acc > best_acc:
                best_acc, best_preds = acc, preds

        del extractor, classifier, discriminator, opt, sch
        cleanup()
        return {"best_acc": float(best_acc), "preds": best_preds}
