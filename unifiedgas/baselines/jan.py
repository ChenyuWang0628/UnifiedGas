"""JAN (Joint Adaptation Networks, Long et al. 2017)

backbone + multi-level cls + Joint MMD over (f1, softmax(o1)) and (f2, softmax(o2)).

JMMD uses the product of Gaussian kernels across layers:
    K_joint((f_s, p_s), (f_t, p_t)) = K_f(f_s, f_t) * K_p(p_s, p_t)

Loss = CE + λ * JMMD.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from itertools import cycle

from ._common import (
    maybe_mixup, cls_loss_with_mix,
    FeatureExtractor, Classifier, make_optim, init_weights,
    make_loaders, evaluate, NUM_CLASSES,
)
from ._common import set_seed, cleanup


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)) -> torch.Tensor:
    """Average multi-bandwidth Gaussian kernel; returns (B_x, B_y)."""
    x_n = F.normalize(x.view(x.size(0), -1), p=2, dim=1)
    y_n = F.normalize(y.view(y.size(0), -1), p=2, dim=1)
    d = torch.cdist(x_n, y_n, p=2) ** 2
    K = sum(torch.exp(-d / (2.0 * s * s)) for s in sigmas) / len(sigmas)
    return K


def joint_mmd(src_layers, tgt_layers) -> torch.Tensor:
    """JMMD over a list of layer pairs (f_l, ...).

    Computes source/target Gaussian kernels at each layer and multiplies them
    element-wise to obtain the joint kernel matrix.
    """
    K_src_src = None
    K_tgt_tgt = None
    K_src_tgt = None
    for s, t in zip(src_layers, tgt_layers):
        Kss = _gaussian_kernel(s, s)
        Ktt = _gaussian_kernel(t, t)
        Kst = _gaussian_kernel(s, t)
        if K_src_src is None:
            K_src_src, K_tgt_tgt, K_src_tgt = Kss, Ktt, Kst
        else:
            K_src_src = K_src_src * Kss
            K_tgt_tgt = K_tgt_tgt * Ktt
            K_src_tgt = K_src_tgt * Kst
    return K_src_src.mean() + K_tgt_tgt.mean() - 2.0 * K_src_tgt.mean()


class JANTrainer:

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.1,
                 device: str = "cuda", lr: float = 1e-3, weight_decay: float = 0.01,
                 lam_jmmd: float = 1.0, dataset: str = "B"):
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.lam_jmmd = lam_jmmd
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
        lam = config.get("lam_jmmd", self.lam_jmmd)

        best_acc, best_preds = 0.0, None
        for ep in range(epochs):
            extractor.train()
            classifier.train()
            for (xs, ys), (xt, _) in zip(src_ld, cycle(tgt_train_ld)):
                xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
                xs, _mix = maybe_mixup(xs, ys, config, src_X.size(0))
                fs = extractor(xs)
                ft = extractor(xt)
                o1_s, o2_s, fo_s = classifier(fs["f1"], fs["f2"])
                o1_t, o2_t, _ = classifier(ft["f1"], ft["f2"])
                cls_l = cls_loss_with_mix(o1_s, o2_s, fo_s, ys, _mix)
                # JMMD: two levels of joint distributions (f, softmax(o))
                jmmd_layer1 = joint_mmd(
                    [fs["f1"], F.softmax(o1_s, dim=1)],
                    [ft["f1"], F.softmax(o1_t, dim=1)])
                jmmd_layer2 = joint_mmd(
                    [fs["f2"], F.softmax(o2_s, dim=1)],
                    [ft["f2"], F.softmax(o2_t, dim=1)])
                jmmd = 0.5 * (jmmd_layer1 + jmmd_layer2)
                loss = cls_l + lam * jmmd
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
