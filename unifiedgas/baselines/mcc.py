"""MCC (Minimum Class Confusion, Jin et al. ECCV 2020) on the shared UnifiedGas backbone.
CE(source, multi-level) + lambda * MCC(target). Single-step target regularization without adversarial alternating updates."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from itertools import cycle

from ._common import (
    maybe_mixup, cls_loss_with_mix,
    FeatureExtractor, Classifier, make_optim, init_weights, make_loaders, evaluate, NUM_CLASSES,
)
from ._common import set_seed, cleanup


def mcc_loss(logits: torch.Tensor, T: float = 2.5) -> torch.Tensor:
    """Minimum Class Confusion loss on target logits (entropy-reweighted)."""
    n, c = logits.shape
    p = F.softmax(logits / T, dim=1)                      # [n, c]
    H = -(p * torch.log(p + 1e-8)).sum(dim=1)             # [n]
    w = 1.0 + torch.exp(-H)
    w = n * w / (w.sum() + 1e-8)                          # Reweight to sum to n
    cov = p.t() @ (w.unsqueeze(1) * p)                    # [c, c] weighted class-correlation matrix
    cov = cov / (cov.sum(dim=1, keepdim=True) + 1e-8)     # Normalize per class
    return (cov.sum() - cov.diagonal().sum()) / c         # Mean off-diagonal class confusion


class MCCTrainer:
    def __init__(self, num_classes: int = NUM_CLASSES, dataset: str = "A",
                 dropout_rate: float = 0.1, lr: float = 1e-3, weight_decay: float = 0.01,
                 mcc_T: float = 2.5):
        self.num_classes = num_classes
        self.dataset = dataset
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.mcc_T = mcc_T

    def train(self, src_X, src_y, tgt_X, tgt_y, device, config: dict, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset, dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        init_weights(extractor)
        init_weights(classifier)
        epochs = config["epochs"]
        bs = config["batch_size"]
        opt, sch = make_optim([extractor, classifier], lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay), total_epochs=epochs)
        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)
        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            p = ep / max(epochs - 1, 1)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0   # Progressive schedule, consistent with other baselines
            extractor.train()
            classifier.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))
                fs = extractor(xs)
                o1, o2, fo = classifier(fs["f1"], fs["f2"])
                cls_l = cls_loss_with_mix(o1, o2, fo, ys, _mix)
                ft = extractor(xt)
                _, _, fot = classifier(ft["f1"], ft["f2"])
                loss = cls_l + lam * mcc_loss(fot, T=self.mcc_T)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
            acc, preds, _ = evaluate(extractor, classifier, tgt_eval_ld, dev)
            if acc > best_acc:
                best_acc, best_preds = acc, preds
        del extractor, classifier, opt, sch
        cleanup()
        return {"best_acc": float(best_acc), "preds": best_preds}
