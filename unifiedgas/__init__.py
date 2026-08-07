"""UnifiedGas: end-to-end unsupervised domain adaptation for drift-robust gas classification.

Reference implementation accompanying the IEEE Sensors Journal submission.
"""

from .models import (
    GasClassifier,
    SensorTransformerBackbone,
    SimpleCNN1D,
    SimpleCNN2D,
    build_backbone,
)
from .losses import (
    mk_mmd, hierarchical_mk_mmd, coral_loss, multi_level_cls_loss, alignment_weight,
)
from .trainer import TrainConfig, UnifiedGasTrainer, set_seed
from .data import (
    load_dataset_a_batch, load_dataset_b_board, load_dataset_b_timeseries,
    extract_steady_state, normalize_pair, apply_norm_stats, make_loaders,
    DATASET_A_CLASSES, DATASET_B_CLASSES,
)

__version__ = "1.0.0"

__all__ = [
    "GasClassifier", "SensorTransformerBackbone", "SimpleCNN1D",
    "SimpleCNN2D", "build_backbone",
    "mk_mmd", "hierarchical_mk_mmd", "coral_loss", "multi_level_cls_loss",
    "alignment_weight",
    "TrainConfig", "UnifiedGasTrainer", "set_seed",
    "load_dataset_a_batch", "load_dataset_b_board", "load_dataset_b_timeseries",
    "extract_steady_state", "normalize_pair", "apply_norm_stats", "make_loaders",
    "DATASET_A_CLASSES", "DATASET_B_CLASSES",
    "__version__",
]
