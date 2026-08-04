"""DSAN (Deep Subdomain Adaptation Network, Zhu et al. IEEE TNNLS 2021).

Unlike MMD/CORAL, which align only marginal distributions, DSAN uses **local MMD
(LMMD)** to align class-wise subdomains. One-hot ground-truth labels weight source
samples, while softmax probabilities weight target samples; the class-wise
weighted MMD terms are then summed. This directly addresses the class-conditional
drift considered here, making DSAN a particularly relevant recent baseline for
Dataset A.

Reference: Zhu, Y. et al., "Deep Subdomain Adaptation Network for Image
Classification," IEEE Transactions on Neural Networks and Learning Systems,
vol. 32, no. 4, pp. 1713-1722, 2021. Uses the shared backbone, multi-level
classifier, and best-selection protocol adopted by the other deep UDA baselines.
"""
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


def _gaussian_kernel(src, tgt, sigmas):
    """Average multi-bandwidth Gaussian kernel, using the UnifiedGas MK-MMD bandwidths."""
    total = torch.cat([src, tgt], dim=0)
    dist = torch.cdist(total, total, p=2) ** 2
    return sum(torch.exp(-dist / (2 * s ** 2)) for s in sigmas) / len(sigmas)


def lmmd_loss(src_f, tgt_f, src_y, tgt_logits, num_classes: int,
              sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)) -> torch.Tensor:
    """Local MMD loss for class-weighted subdomain alignment.

    Source weights use one-hot ground-truth labels, while target weights use
    softmax pseudo-probabilities. Per-class weights are normalized separately
    to construct the ss, tt, and st weight matrices.
    """
    bs_s, bs_t = src_f.size(0), tgt_f.size(0)
    if bs_s == 0 or bs_t == 0:
        return torch.zeros((), device=src_f.device)

    s_onehot = F.one_hot(src_y, num_classes).float()          # [Bs, C]
    t_soft = F.softmax(tgt_logits, dim=1)                      # [Bt, C]

    # Normalize per class and retain classes with mass in both domains
    s_sum = s_onehot.sum(dim=0, keepdim=True)                  # [1, C]
    t_sum = t_soft.sum(dim=0, keepdim=True)
    valid = ((s_sum > 0) & (t_sum > 0)).float()                # [1, C]
    n_valid = valid.sum()
    if n_valid < 1:
        return torch.zeros((), device=src_f.device)

    s_w = s_onehot / (s_sum + 1e-8) * valid                    # [Bs, C]
    t_w = t_soft / (t_sum + 1e-8) * valid                      # [Bt, C]

    W_ss = s_w @ s_w.t()                                       # [Bs, Bs]
    W_tt = t_w @ t_w.t()                                       # [Bt, Bt]
    W_st = s_w @ t_w.t()                                       # [Bs, Bt]

    s_n = F.normalize(src_f.view(bs_s, -1), p=2, dim=1)
    t_n = F.normalize(tgt_f.view(bs_t, -1), p=2, dim=1)
    K = _gaussian_kernel(s_n, t_n, sigmas)
    Kss = K[:bs_s, :bs_s]
    Ktt = K[bs_s:, bs_s:]
    Kst = K[:bs_s, bs_s:]

    loss = (W_ss * Kss).sum() + (W_tt * Ktt).sum() - 2 * (W_st * Kst).sum()
    return loss / n_valid


class DSANTrainer:
    """DSAN: source CE + class-wise subdomain alignment with LMMD."""

    def __init__(self, num_classes: int = NUM_CLASSES, dataset: str = "A",
                 dropout_rate: float = 0.1, lr: float = 1e-3, weight_decay: float = 0.01,
                 lam_lmmd: float = 1.0):
        self.num_classes = num_classes
        self.dataset = dataset
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.lam_lmmd = lam_lmmd

    def train(self, src_X, src_y, tgt_X, tgt_y, device, config: dict, seed: int = 42) -> dict:
        set_seed(seed)
        dev = torch.device(device)
        extractor = FeatureExtractor(dataset=self.dataset, dropout_rate=self.dropout_rate).to(dev)
        classifier = Classifier(self.num_classes).to(dev)
        init_weights(extractor)
        init_weights(classifier)

        epochs = config["epochs"]
        bs = config["batch_size"]
        lam_lmmd = config.get("lam_lmmd", self.lam_lmmd)
        opt, sch = make_optim([extractor, classifier], lr=config.get("lr", self.lr),
                              weight_decay=config.get("weight_decay", self.weight_decay),
                              total_epochs=epochs)
        src_ld, tgt_train_ld, tgt_eval_ld = make_loaders(src_X, src_y, tgt_X, tgt_y, bs)

        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            p = ep / max(epochs - 1, 1)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0   # Same λ schedule as the other baselines
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
                # Use pre-mixup ground-truth labels as source weights; with mixup, use labels before permutation
                y_for_lmmd = ys if _mix is None else _mix[0]
                lmmd = lmmd_loss(fs["f1"], ft["f1"], y_for_lmmd, fot.detach(),
                                 self.num_classes)

                loss = cls_l + lam * lam_lmmd * lmmd
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
