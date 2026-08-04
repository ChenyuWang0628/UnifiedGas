#!/usr/bin/env python3
"""Train one of the deep UDA baselines on a single source -> target task.

Every baseline shares the UnifiedGas backbone, classifier heads, logit fusion,
classification loss, optimizer, and schedule; only the adaptation objective
differs.  Running this script and ``train.py`` under the same settings therefore
compares adaptation objectives rather than network capacity.

Examples
--------
    # DSAN on Dataset A, Setting 1, Batch 1 -> Batch 6
    python scripts/train_baseline.py --method dsan --dataset A \
        --source 1 --target 6 --epochs 400 --seeds 42,123,2024

    # List the available baselines
    python scripts/train_baseline.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unifiedgas import (  # noqa: E402
    load_dataset_a_batch, load_dataset_b_board, normalize_pair,
)
from unifiedgas.baselines import TRAINER_REGISTRY  # noqa: E402


def load_pair(dataset: str, data_dir: Path, src: int, tgt: int, norm: str):
    """Load a source/target pair under the same protocol as ``train.py``."""
    if dataset.upper() == "A":
        sx, sy = load_dataset_a_batch(data_dir / f"batch{src}.dat")
        tx, ty = load_dataset_a_batch(data_dir / f"batch{tgt}.dat")
        return sx, sy, tx, ty
    sx, sy = load_dataset_b_board(data_dir / f"batch{src}.csv")
    tx, ty = load_dataset_b_board(data_dir / f"batch{tgt}.csv")
    sxn, txn = normalize_pair(sx.numpy(), tx.numpy(), mode=norm)
    return (torch.from_numpy(sxn.astype(np.float32)), sy,
            torch.from_numpy(txn.astype(np.float32)), ty)


def main():
    ap = argparse.ArgumentParser(
        description="Train a deep UDA baseline on the shared backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="list available baselines and exit")
    ap.add_argument("--method", choices=sorted(TRAINER_REGISTRY))
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--source", type=int, help="source batch / board")
    ap.add_argument("--target", type=int, help="target batch / board")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--norm", default="per_batch",
                    choices=["per_batch", "source_only", "global", "none"],
                    help="normalization protocol (Dataset B only)")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="optional JSON results path")
    args = ap.parse_args()

    if args.list:
        print("available baselines:")
        for k in sorted(TRAINER_REGISTRY):
            print(f"  {k}")
        return

    missing = [f"--{n}" for n in ("method", "source", "target")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    dataset = args.dataset.upper()
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data" / f"DataSet{dataset}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    num_classes = 6 if dataset == "A" else 4

    print(f"{args.method.upper()} | Dataset {dataset} | "
          f"batch/board {args.source} -> {args.target}")
    print(f"  data      : {data_dir}")
    print(f"  device    : {device}   epochs: {args.epochs}   seeds: {seeds}")

    sx, sy, tx, ty = load_pair(dataset, data_dir, args.source, args.target, args.norm)
    print(f"  source    : {tuple(sx.shape)}  target: {tuple(tx.shape)}")

    config = {"epochs": args.epochs, "batch_size": args.batch_size}
    accs = []
    for seed in seeds:
        t0 = time.time()
        trainer = TRAINER_REGISTRY[args.method](num_classes=num_classes, dataset=dataset)
        res = trainer.train(sx, sy, tx, ty, device, config, seed=seed)
        accs.append(res["best_acc"])
        print(f"  seed {seed:>5}: {res['best_acc'] * 100:.2f}%  ({time.time() - t0:.0f}s)")

    print(f"\n{'=' * 56}")
    print(f"{args.method} mean over {len(seeds)} seed(s): "
          f"{np.mean(accs) * 100:.2f}% +/- {np.std(accs) * 100:.2f}")
    print(f"{'=' * 56}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "method": args.method, "dataset": dataset,
            "source": args.source, "target": args.target,
            "epochs": args.epochs, "seeds": seeds,
            "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "per_seed": [float(a) for a in accs],
        }, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
