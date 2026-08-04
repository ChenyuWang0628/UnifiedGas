"""MDD (Margin Disparity Discrepancy, Zhang et al. ICML 2019) on the shared UnifiedGas backbone.

The main classifier minimizes CE on source samples. An adversarial auxiliary
classifier is connected to the features through the GRL:
- source: adv matches the main classifier -> CE(adv_s, argmax(main_s))
- target: adv diverges from the main classifier -> NLL(shift_log(1 - softmax(adv_t)), argmax(main_t))
- transfer = margin * src_disparity + tgt_disparity (the GRL makes the feature extractor maximize in the opposite direction)

Reference: Zhang, Y. et al., "Bridging Theory and Algorithm for Domain
Adaptation," ICML 2019. Uses the shared backbone, multi-level classifier, and
best-selection protocol adopted by the other deep UDA baselines."""
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


def _shift_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """log(clamp(x, min=eps)); prevents log(0), as in tllib."""
    return torch.log(torch.clamp(x, min=eps))


class AdvClassifier(nn.Module):
    """Adversarial auxiliary classifier: f1 (128) -> hidden -> num_classes, matching the main head capacity."""

    def __init__(self, num_classes: int, in_dim: int = 128, hidden: int = 128,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MDDTrainer:
    def __init__(self, num_classes: int = NUM_CLASSES, dataset: str = "A",
                 dropout_rate: float = 0.1, lr: float = 1e-3, weight_decay: float = 0.01,
                 margin: float = 4.0):
        self.num_classes = num_classes
        self.dataset = dataset
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.margin = margin

    def train(self, src_X, src_y, tgt_X, tgt_y, device, config: dict, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset, dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        adv = AdvClassifier(self.num_classes, in_dim=128, dropout_rate=self.dropout_rate).to(dev)
        init_weights(extractor)
        init_weights(classifier)
        init_weights(adv)
        epochs = config["epochs"]
        bs = config["batch_size"]
        opt, sch = make_optim([extractor, classifier, adv], lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay), total_epochs=epochs)
        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)
        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            p = ep / max(epochs - 1, 1)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0   # Progressive GRL coefficient
            extractor.train()
            classifier.train()
            adv.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))
                fs = extractor(xs)
                ft = extractor(xt)
                o1s, o2s, fos = classifier(fs["f1"], fs["f2"])
                o1t, o2t, fot = classifier(ft["f1"], ft["f2"])
                cls_l = cls_loss_with_mix(o1s, o2s, fos, ys, _mix)

                # Detached source/target hard labels from the main classifier serve as adv targets
                ps = fos.argmax(1).detach()
                pt = fot.argmax(1).detach()
                # GRL reverses feature gradients: the extractor maximizes while adv minimizes
                adv_s = adv(grad_reverse(fs["f1"], lam))
                adv_t = adv(grad_reverse(ft["f1"], lam))
                src_disp = F.cross_entropy(adv_s, ps)
                tgt_disp = F.nll_loss(_shift_log(1.0 - F.softmax(adv_t, dim=1)), pt)
                transfer = (self.margin * src_disp + tgt_disp) / (1.0 + self.margin)

                loss = cls_l + transfer
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
            acc, preds, _ = evaluate(extractor, classifier, tgt_eval_ld, dev)
            if acc > best_acc:
                best_acc, best_preds = acc, preds
        del extractor, classifier, adv, opt, sch
        cleanup()
        return {"best_acc": float(best_acc), "preds": best_preds}
