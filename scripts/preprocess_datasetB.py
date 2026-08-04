#!/usr/bin/env python3
"""Convert the raw Twin Gas Sensor Arrays recordings into model inputs.

Two representations are produced, matching the two protocols in the paper:

steady-state (default)
    Per-sensor mean over the last 5,000 samples (the final 50 s of each 600 s
    exposure at 100 Hz), giving one 8-dimensional vector per recording.
    Written as ``batch1.csv ... batch5.csv`` with rows ``f0,...,f7,label``.

raw time series (``--timeseries``)
    Each recording downsampled to ``T`` timesteps by indexing ``T`` uniformly
    spaced positions over that recording's own length.  Written as a single
    ``.npz`` holding ``board{i}_X`` of shape ``(N_i, T, 8)`` and ``board{i}_y``.

Usage
-----
    python preprocess_datasetB.py --data_dir data/DataSetB-raw --out_dir data/DataSetB
    python preprocess_datasetB.py --data_dir data/DataSetB-raw --timeseries --T 256 \
        --out_dir data/cache
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np

GAS_TO_LABEL = {"GCO": 0, "GEa": 1, "GEy": 2, "GMe": 3}
FNAME_RE = re.compile(r"^B([1-5])_(GCO|GEa|GEy|GMe)_F(\d{3})_R(\d+)\.txt$")

STEADY_STATE_WINDOW = 5000  # final 50 s at 100 Hz
N_SENSORS = 8


def iter_recordings(data_dir: Path):
    """Yield ``(board, label, path)`` for every correctly named recording."""
    files = sorted(glob.glob(str(data_dir / "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"no .txt recordings under {data_dir}. Download the UCI Twin Gas "
            f"Sensor Arrays dataset first (see docs/DATA.md)."
        )
    n_skipped = 0
    for fp in files:
        m = FNAME_RE.match(os.path.basename(fp))
        if m is None:
            n_skipped += 1
            continue
        yield int(m.group(1)), GAS_TO_LABEL[m.group(2)], fp
    if n_skipped:
        print(f"  [warn] skipped {n_skipped} file(s) with unexpected names")


def build_steady_state(data_dir: Path, window: int):
    boards: dict[int, dict[str, list]] = {b: {"X": [], "y": []} for b in range(1, 6)}
    for board, label, fp in iter_recordings(data_dir):
        sensors = np.loadtxt(fp, dtype=np.float64)[:, 1 : N_SENSORS + 1]
        if sensors.shape[0] < window:
            raise ValueError(f"{fp}: {sensors.shape[0]} samples < window {window}")
        boards[board]["X"].append(sensors[-window:].mean(axis=0))
        boards[board]["y"].append(label)
    return {b: (np.asarray(v["X"], dtype=np.float32),
                np.asarray(v["y"], dtype=np.int64))
            for b, v in boards.items()}


def build_timeseries(data_dir: Path, T: int):
    boards: dict[int, dict[str, list]] = {b: {"X": [], "y": []} for b in range(1, 6)}
    for board, label, fp in iter_recordings(data_dir):
        sig = np.loadtxt(fp, dtype=np.float32)[:, 1 : N_SENSORS + 1]
        # Index T positions over this recording's own length, so truncated runs
        # still yield exactly T timesteps.
        idx = np.linspace(0, sig.shape[0] - 1, T).astype(np.int64)
        boards[board]["X"].append(sig[idx])
        boards[board]["y"].append(label)
    return {b: (np.stack(v["X"]).astype(np.float32),
                np.asarray(v["y"], dtype=np.int64))
            for b, v in boards.items()}


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess the raw Twin Gas Sensor Arrays recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", required=True,
                    help="directory holding the raw B*_G*_F*_R*.txt recordings")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--timeseries", action="store_true",
                    help="produce the raw time-series representation instead")
    ap.add_argument("--T", type=int, default=256, help="timesteps (time-series mode)")
    ap.add_argument("--window", type=int, default=STEADY_STATE_WINDOW,
                    help="steady-state window in samples (steady-state mode)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.timeseries:
        print(f"Building raw time-series inputs (T={args.T}) from {data_dir}")
        boards = build_timeseries(data_dir, args.T)
        payload = {}
        for b, (X, y) in boards.items():
            payload[f"board{b}_X"] = X
            payload[f"board{b}_y"] = y
            print(f"  board {b}: X={X.shape}  y={y.shape}  "
                  f"class counts={np.bincount(y, minlength=4).tolist()}")
        out = out_dir / f"datasetB_ts_T{args.T}.npz"
        np.savez_compressed(out, **payload)
        print(f"[saved] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"Building steady-state features (window={args.window}) from {data_dir}")
    boards = build_steady_state(data_dir, args.window)
    for b, (X, y) in boards.items():
        out = out_dir / f"batch{b}.csv"
        # %.10g keeps every digit a float32 can represent.  With a fixed 8-decimal
        # format the written features differ from the in-memory ones by ~4e-6,
        # which is enough to move a borderline MLP decision boundary and make the
        # from-features route disagree with the from-raw-data route.
        np.savetxt(out, np.hstack([X, y.reshape(-1, 1).astype(np.float32)]),
                   delimiter=",", fmt="%.10g")
        print(f"  board {b}: X={X.shape}  y={y.shape}  "
              f"class counts={np.bincount(y, minlength=4).tolist()}  -> {out}")
    print(f"[done] wrote 5 board files to {out_dir}")


if __name__ == "__main__":
    main()
