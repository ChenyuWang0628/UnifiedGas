"""Deep UDA baselines re-implemented on the UnifiedGas backbone.

All eight baselines share the *same* feature extractor, classifier heads, logit
fusion, classification loss, optimizer, and learning-rate schedule as
UnifiedGas.  Only the adaptation objective differs, so a comparison between them
measures the objective rather than network capacity.

============  =====================================  ==========================
Baseline      Adaptation objective                   Reference
============  =====================================  ==========================
DANN          adversarial domain discriminator       Ganin et al., JMLR 2016
Deep CORAL    second-order covariance alignment      Sun & Saenko, ECCVW 2016
JAN           joint MMD over multiple layers         Long et al., ICML 2017
CDAN          conditional adversarial alignment      Long et al., NeurIPS 2018
MDD           margin disparity discrepancy           Zhang et al., ICML 2019
MCC           minimum class confusion                Jin et al., ECCV 2020
DSAN          local MMD over class subdomains        Zhu et al., TNNLS 2021
SDAT          sharpness-aware adversarial training   Rangwani et al., ICML 2022
============  =====================================  ==========================

Every trainer exposes the same interface::

    trainer = DANNTrainer(num_classes=6, dataset="A")
    result = trainer.train(src_X, src_y, tgt_X, tgt_y, device,
                           {"epochs": 400, "batch_size": 8}, seed=42)
    result["best_acc"]   # target accuracy at the reported epoch
"""

from .dann import DANNTrainer
from .deep_coral import DeepCORALTrainer
from .jan import JANTrainer
from .cdan import CDANTrainer
from .mcc import MCCTrainer
from .mdd import MDDTrainer
from .dsan import DSANTrainer
from .sdat import SDATTrainer

TRAINER_REGISTRY = {
    "dann": DANNTrainer,
    "deep_coral": DeepCORALTrainer,
    "jan": JANTrainer,
    "cdan": CDANTrainer,
    "mcc": MCCTrainer,
    "mdd": MDDTrainer,
    "dsan": DSANTrainer,
    "sdat": SDATTrainer,
}

__all__ = [
    "DANNTrainer", "DeepCORALTrainer", "JANTrainer", "CDANTrainer",
    "MCCTrainer", "MDDTrainer", "DSANTrainer", "SDATTrainer",
    "TRAINER_REGISTRY",
]
