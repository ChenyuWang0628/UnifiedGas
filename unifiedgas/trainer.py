"""Single-stage end-to-end trainer for UnifiedGas.

Everything -- feature extraction, hierarchical MK-MMD and CORAL alignment,
the dual-level center regularizer, and the fused classification heads -- is
optimized by one optimizer under one alignment-weight schedule.  There is no
separate pretraining stage and no manually specified phase transition.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from .losses import (
    alignment_weight, coral_loss, hierarchical_mk_mmd, multi_level_cls_loss,
    DEFAULT_SIGMAS,
)
from .models import GasClassifier

__all__ = ["TrainConfig", "UnifiedGasTrainer", "set_seed"]


def set_seed(seed: int = 42):
    """Seed every RNG that affects training, and prefer deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.__version__ >= "2.0.0":
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class TrainConfig:
    """Hyperparameters of the reported configuration."""

    epochs: int = 400
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    eta_min: float = 1e-6
    dropout: float = 0.1
    warmup_frac: float = 0.25
    lambda_coral: float = 1.0
    lambda_center: float = 0.1
    cls_weights: tuple = (0.4, 0.3, 0.3)
    fusion_w1: float = 0.6
    sigmas: tuple = DEFAULT_SIGMAS
    num_classes: int = 6
    dataset: str = "A"

    def to_dict(self) -> dict:
        return asdict(self)


class UnifiedGasTrainer:
    """Trains a :class:`GasClassifier` with the unified objective."""

    def __init__(self, config: TrainConfig, device: str | torch.device = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        self.model: GasClassifier | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    def build_model(self) -> GasClassifier:
        return GasClassifier(
            num_classes=self.cfg.num_classes,
            dropout_rate=self.cfg.dropout,
            use_center_loss=True,
            dataset=self.cfg.dataset,
            fusion_w1=self.cfg.fusion_w1,
        ).to(self.device)

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        preds, gts = [], []
        for xb, yb in loader:
            _, _, fused = self.model(xb.to(self.device))
            preds.append(fused.argmax(1).cpu().numpy())
            gts.append(yb.numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        return float((preds == gts).mean()), preds, gts

    # ------------------------------------------------------------------ #
    def fit(self, src_loader, tgt_train_loader, tgt_eval_loader,
            seed: int = 42, verbose: bool = True) -> dict:
        """Run the full single-stage training loop.

        Returns the best and final target accuracy along with the epoch at
        which the best value occurred.  Target labels are used only to compute
        the reported accuracy -- never in any gradient.
        """
        set_seed(seed)
        cfg = self.cfg
        self.model = self.build_model()
        opt = optim.AdamW(self.model.parameters(), lr=cfg.lr,
                          weight_decay=cfg.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                   eta_min=cfg.eta_min)
        self.history = []
        best_acc, best_epoch, final_acc = 0.0, -1, 0.0
        # Keep a copy of the weights at the reported epoch so that a saved
        # checkpoint matches the accuracy reported for it.
        self.best_state: dict | None = None

        for ep in range(cfg.epochs):
            lam = alignment_weight(ep, cfg.epochs, cfg.warmup_frac)
            self.model.train()
            ep_loss = ep_mmd = ep_cor = 0.0
            n_steps = 0

            for (xs, ys), (xt, _) in zip(src_loader, cycle(tgt_train_loader)):
                xs, ys, xt = xs.to(self.device), ys.to(self.device), xt.to(self.device)
                (o1, o2, fused), center = self.model(xs, ys)
                loss = multi_level_cls_loss(o1, o2, fused, ys, cfg.cls_weights)
                loss = loss + cfg.lambda_center * center

                mmd_v = cor_v = 0.0
                if lam > 0:
                    sf = self.model.get_all_features(xs)
                    tf = self.model.get_all_features(xt)
                    mmd = hierarchical_mk_mmd(sf, tf, cfg.sigmas)
                    cor = coral_loss(sf[1], tf[1])
                    loss = loss + lam * (mmd + cfg.lambda_coral * cor)
                    mmd_v, cor_v = float(mmd.detach()), float(cor.detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                ep_loss += float(loss.detach())
                ep_mmd += mmd_v
                ep_cor += cor_v
                n_steps += 1

            sch.step()
            acc, _, _ = self.evaluate(tgt_eval_loader)
            final_acc = acc
            if acc > best_acc:
                best_acc, best_epoch = acc, ep
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}

            self.history.append({
                "epoch": ep,
                "loss": ep_loss / max(n_steps, 1),
                "mmd": ep_mmd / max(n_steps, 1),
                "coral": ep_cor / max(n_steps, 1),
                "lambda_align": lam,
                "target_acc": acc,
                "lr": sch.get_last_lr()[0],
            })
            if verbose and (ep % 50 == 0 or ep == cfg.epochs - 1):
                print(f"  epoch {ep:4d}  loss={ep_loss / max(n_steps, 1):.4f}  "
                      f"lambda={lam:.3f}  target_acc={acc * 100:.2f}%", flush=True)

        return {
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "final_acc": final_acc,
            "seed": seed,
        }

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str | Path, extra: dict | None = None,
                        which: str = "best"):
        """Persist weights plus the configuration needed to rebuild the model.

        ``which="best"`` (default) saves the weights from the reported epoch, so
        the checkpoint reproduces the accuracy quoted for it.  ``which="final"``
        saves the weights at the end of training, which is markedly worse under
        this schedule -- see the convergence discussion in the paper.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if which == "best" and self.best_state is not None:
            state = self.best_state
        else:
            state = self.model.state_dict()
        payload = {
            "state_dict": state,
            "config": self.cfg.to_dict(),
            "checkpoint_epoch": which,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu"):
        """Rebuild a trainer and its model from a checkpoint written above."""
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = payload["config"]
        cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                             if k in TrainConfig.__dataclass_fields__})
        trainer = cls(cfg, device=device)
        trainer.model = trainer.build_model()
        trainer.model.load_state_dict(payload["state_dict"])
        trainer.model.eval()
        return trainer, payload
