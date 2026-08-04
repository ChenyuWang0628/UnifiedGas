"""Deep CORAL (Sun & Saenko 2016)

backbone + multi-level cls + CORAL distance on f1::

    L_CORAL = ||cov(src_f1) - cov(tgt_f1)||_F^2 / (4 * d^2)

The differentiable PyTorch implementation follows Deep CORAL (Sun & Saenko, ECCV 2016 Workshops).
"""

from __future__ import annotations

import torch
from itertools import cycle

from ._common import (
    maybe_mixup, cls_loss_with_mix,
    FeatureExtractor, Classifier, make_optim, init_weights,
    make_loaders, evaluate, NUM_CLASSES,
)
from ._common import set_seed, cleanup


def coral_loss(src_f: torch.Tensor, tgt_f: torch.Tensor) -> torch.Tensor:
    """CORAL: Frobenius distance between covariance matrices, normalized by 4d^2.

    Reference: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain
    Adaptation," ECCV 2016 Workshops.
    """
    d = src_f.size(1)
    ns, nt = src_f.size(0), tgt_f.size(0)

    src_c = src_f - src_f.mean(0, keepdim=True)
    tgt_c = tgt_f - tgt_f.mean(0, keepdim=True)

    cov_s = src_c.t() @ src_c / max(ns - 1, 1)
    cov_t = tgt_c.t() @ tgt_c / max(nt - 1, 1)

    diff = cov_s - cov_t
    return (diff * diff).sum() / (4.0 * d * d)


class DeepCORALTrainer:

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.1,
                 device: str = "cuda", lr: float = 1e-3, weight_decay: float = 0.01,
                 lam_coral: float = 1.0, dataset: str = "B"):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.lam_coral = lam_coral
        self.dataset = dataset

    def train(self, src_X, src_y, tgt_X, tgt_y, device, config, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset,
                                       dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        init_weights(extractor)
        init_weights(classifier)

        epochs = config["epochs"]
        bs = config["batch_size"]
        opt, sch = make_optim([extractor, classifier],
                              lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay),
                              total_epochs=epochs)
        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)
        lam = config.get("lam_coral", self.lam_coral)

        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            extractor.train()
            classifier.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))
                fs = extractor(xs)
                ft = extractor(xt)
                o1, o2, fo = classifier(fs["f1"], fs["f2"])
                cls_l = cls_loss_with_mix(o1, o2, fo, ys, _mix)
                cor_l = coral_loss(fs["f1"], ft["f1"])
                loss = cls_l + lam * cor_l
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
