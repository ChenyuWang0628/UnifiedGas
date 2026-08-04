"""Dataset loading and normalization for both benchmarks.

Dataset A -- UCI Gas Sensor Array Drift
    Ten batches in LIBSVM-like format.  Every sample carries 128 features
    (16 sensors x 8 features) and one of six gas labels.

Dataset B -- UCI Twin Gas Sensor Arrays
    Five replicate boards, four gases.  Two input representations are
    supported: the 8-D steady-state vector (per-sensor mean over the final
    5,000 samples of each exposure) and the raw transient time series,
    uniformly downsampled to a fixed number of timesteps.

Normalization protocols
-----------------------
``per_batch``  z-score every domain with its own feature-wise statistics.
               Uses unlabeled target statistics only, never target labels.
               This is the protocol used for all trained models in the paper.
``source_only`` fit on the source domain, apply to both.
``global``     fit once on the pooled data of all domains.
``none``       no normalization.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "load_dataset_a_batch",
    "load_dataset_b_board",
    "load_dataset_b_timeseries",
    "extract_steady_state",
    "normalize_pair",
    "apply_norm_stats",
    "make_loaders",
    "DATASET_A_CLASSES",
    "DATASET_B_CLASSES",
]

DATASET_A_CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
DATASET_B_CLASSES = ["CO", "Ethanol", "Ethylene", "Methane"]

_B_GAS_TO_LABEL = {"GCO": 0, "GEa": 1, "GEy": 2, "GMe": 3}
_B_FNAME_RE = re.compile(r"^B([1-5])_(GCO|GEa|GEy|GMe)_F(\d{3})_R(\d+)\.txt$")

STEADY_STATE_WINDOW = 5000  # final 50 s of each 600 s exposure at 100 Hz


# --------------------------------------------------------------------------- #
# Dataset A
# --------------------------------------------------------------------------- #
def load_dataset_a_batch(path: str | Path, num_sensors: int = 16,
                         num_features: int = 8, standardize: bool = True):
    """Load one Dataset-A batch file into ``(X, y)`` tensors.

    Each line is ``<label>;<concentration> 1:<v> 2:<v> ...`` with 128 indexed
    features.  The features are reshaped to ``(16, 8)`` so that the 2-D
    backbone sees sensors along one axis and features along the other.
    """
    df = pd.read_csv(path, delimiter=";", header=None)
    data, labels = [], []
    for _, row in df.iterrows():
        labels.append(int(row[0]) - 1)                    # labels are 1-indexed
        feats = [[0.0] * num_features for _ in range(num_sensors)]
        for item in " ".join(str(row[1]).split()[1:]).split():
            if ":" not in item:
                continue
            sf, val = item.split(":")
            sid, fid = divmod(int(sf) - 1, num_features)
            feats[sid][fid] = float(val)
        data.append(feats)

    X = np.asarray(data, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if standardize:
        n, s, f = X.shape
        X = StandardScaler().fit_transform(X.reshape(n, -1)).reshape(n, s, f)
    X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)   # (N, 1, 16, 8)
    return X_t, torch.from_numpy(y)


# --------------------------------------------------------------------------- #
# Dataset B -- steady-state representation
# --------------------------------------------------------------------------- #
def extract_steady_state(path: str | Path, window: int = STEADY_STATE_WINDOW) -> np.ndarray:
    """Per-sensor mean over the last ``window`` samples of one raw recording.

    Column 0 of the file holds the timestamp and is dropped; columns 1-8 are
    the eight MOX sensor channels sampled at 100 Hz.
    """
    data = np.loadtxt(path, dtype=np.float64)
    sensors = data[:, 1:9]
    if sensors.shape[0] < window:
        raise ValueError(f"{path}: only {sensors.shape[0]} samples, need >= {window}")
    return sensors[-window:].mean(axis=0).astype(np.float32)


def load_dataset_b_board(path: str | Path, num_features: int = 8):
    """Load one preprocessed Dataset-B board CSV (8 features + label column)."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    X = arr[:, :num_features].astype(np.float32)
    y = arr[:, num_features].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def build_dataset_b_from_raw(data_dir: str | Path, window: int = STEADY_STATE_WINDOW):
    """Extract steady-state features for all five boards straight from raw .txt."""
    files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        raise FileNotFoundError(f"no raw recordings under {data_dir}")
    boards: dict[int, dict[str, list]] = {b: {"X": [], "y": []} for b in range(1, 6)}
    for fp in files:
        m = _B_FNAME_RE.match(os.path.basename(fp))
        if m is None:
            continue
        boards[int(m.group(1))]["X"].append(extract_steady_state(fp, window))
        boards[int(m.group(1))]["y"].append(_B_GAS_TO_LABEL[m.group(2)])
    return {
        b: (np.asarray(v["X"], dtype=np.float32), np.asarray(v["y"], dtype=np.int64))
        for b, v in boards.items()
    }


# --------------------------------------------------------------------------- #
# Dataset B -- raw time-series representation
# --------------------------------------------------------------------------- #
def load_dataset_b_timeseries(cache_path: str | Path, board: int):
    """Load a cached ``(N, T, 8)`` time-series tensor for one board."""
    z = np.load(cache_path)
    X = z[f"board{board}_X"].astype(np.float32)
    y = z[f"board{board}_y"].astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def build_dataset_b_timeseries(data_dir: str | Path, T: int = 256):
    """Downsample every raw recording to ``T`` timesteps.

    The recordings are nominally 60,000 samples (600 s at 100 Hz); a few runs
    stop early.  ``np.linspace`` therefore indexes each recording over its own
    actual length, so every sample yields exactly ``T`` timesteps regardless of
    truncation.
    """
    files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    boards: dict[int, dict[str, list]] = {b: {"X": [], "y": []} for b in range(1, 6)}
    for fp in files:
        m = _B_FNAME_RE.match(os.path.basename(fp))
        if m is None:
            continue
        arr = np.loadtxt(fp, dtype=np.float32)
        sig = arr[:, 1:9]
        idx = np.linspace(0, sig.shape[0] - 1, T).astype(np.int64)
        boards[int(m.group(1))]["X"].append(sig[idx])
        boards[int(m.group(1))]["y"].append(_B_GAS_TO_LABEL[m.group(2)])
    return {
        b: (np.stack(v["X"]).astype(np.float32), np.asarray(v["y"], dtype=np.int64))
        for b, v in boards.items()
    }


# --------------------------------------------------------------------------- #
# Normalization and loaders
# --------------------------------------------------------------------------- #
def normalize_pair(src: np.ndarray, tgt: np.ndarray, mode: str = "per_batch",
                   return_stats: bool = False):
    """Normalize a source/target pair under one of the supported protocols.

    With ``return_stats=True`` the fitted mean/scale are also returned, as
    ``None`` for ``per_batch`` and ``none`` (which fit nothing that transfers to
    unseen data) and as a ``(mean, scale)`` pair otherwise.  Storing those
    statistics in the checkpoint lets inference reapply the exact transform the
    model was trained under instead of refitting on whatever it is shown.
    """
    if mode == "none":
        return (src, tgt, None) if return_stats else (src, tgt)
    if mode == "per_batch":
        out = (StandardScaler().fit_transform(src),
               StandardScaler().fit_transform(tgt))
        return (*out, None) if return_stats else out
    if mode == "source_only":
        sc = StandardScaler().fit(src)
    elif mode == "global":
        sc = StandardScaler().fit(np.concatenate([src, tgt], axis=0))
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    out = (sc.transform(src), sc.transform(tgt))
    if return_stats:
        return (*out, (sc.mean_.astype(np.float64), sc.scale_.astype(np.float64)))
    return out


def apply_norm_stats(X: np.ndarray, stats, mode: str = "per_batch") -> np.ndarray:
    """Reapply a training-time normalization protocol to new data.

    ``source_only`` and ``global`` carry stored ``(mean, scale)`` statistics and
    must reuse them; ``per_batch`` fits on the incoming domain, because fitting
    per domain *is* that protocol; ``none`` leaves the data untouched.  Passing
    ``stats=None`` for a protocol that needs statistics is an error rather than a
    silent fallback, since refitting would change the transform the model saw.
    """
    if mode == "none":
        return X
    if mode == "per_batch":
        return StandardScaler().fit_transform(X)
    if stats is None:
        raise ValueError(
            f"normalization protocol '{mode}' requires the statistics fitted "
            "during training, but the checkpoint does not carry them. Retrain "
            "with the current scripts/train.py, or pass --norm per_batch."
        )
    mean, scale = stats
    return (X - np.asarray(mean)) / np.asarray(scale)


def make_loaders(src_X, src_y, tgt_X, tgt_y, batch_size: int = 8):
    """Build the source-train / target-train / target-eval loaders.

    ``drop_last=True`` on the training loaders prevents a final batch of size 1
    from breaking BatchNorm; the eval loader keeps every sample.
    """
    src_ld = DataLoader(TensorDataset(src_X, src_y), batch_size=batch_size,
                        shuffle=True, drop_last=True)
    tgt_ds = TensorDataset(tgt_X, tgt_y)
    tgt_train_ld = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_eval_ld = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False)
    return src_ld, tgt_train_ld, tgt_eval_ld
