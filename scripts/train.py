#!/usr/bin/env python3
"""Train UnifiedGas on one source -> target transfer task.

Examples
--------
    # Dataset A, Setting 1 (Batch 1 is the fixed source), Batch 1 -> Batch 6
    python scripts/train.py --dataset A --source 1 --target 6 \
        --data_dir data/DataSetA --epochs 400 --seeds 42,123,2024 \
        --save_checkpoint checkpoints/unifiedgas_A_b1b6.pt

    # Dataset B, steady-state features, Board 1 -> Board 2
    python scripts/train.py --dataset B --source 1 --target 2 \
        --data_dir data/DataSetB --epochs 400 --seeds 42

Target labels never enter a gradient update. They are used only to score each
epoch, and the reported accuracy is the one at the best-scoring epoch -- the
retrospective oracle-selection protocol the paper applies uniformly to every
deep method. The saved checkpoint holds that epoch's weights, so it too depends
on target labels. The final-epoch accuracy is printed alongside as the
conservative alternative.
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
    TrainConfig, UnifiedGasTrainer, load_dataset_a_batch, load_dataset_b_board,
    make_loaders, normalize_pair,
)


def load_pair(dataset: str, data_dir: Path, src: int, tgt: int, norm: str):
    """Load and normalize one source/target pair.

    Returns the fitted normalization statistics alongside the tensors so that
    they can be stored in the checkpoint and reused verbatim at inference time.
    """
    if dataset.upper() == "A":
        # Dataset A files are z-scored per file inside the loader, matching the
        # per-batch protocol used throughout the paper.
        sx, sy = load_dataset_a_batch(data_dir / f"batch{src}.dat")
        tx, ty = load_dataset_a_batch(data_dir / f"batch{tgt}.dat")
        return sx, sy, tx, ty, None
    sx, sy = load_dataset_b_board(data_dir / f"batch{src}.csv")
    tx, ty = load_dataset_b_board(data_dir / f"batch{tgt}.csv")
    sxn, txn, stats = normalize_pair(sx.numpy(), tx.numpy(), mode=norm,
                                     return_stats=True)
    return (torch.from_numpy(sxn.astype(np.float32)), sy,
            torch.from_numpy(txn.astype(np.float32)), ty, stats)


def main():
    ap = argparse.ArgumentParser(
        description="Train UnifiedGas on one transfer task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", default="A", choices=["A", "B"])
    ap.add_argument("--data_dir", default=None,
                    help="defaults to data/DataSetA or data/DataSetB")
    ap.add_argument("--source", type=int, required=True, help="source batch / board")
    ap.add_argument("--target", type=int, required=True, help="target batch / board")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_coral", type=float, default=1.0)
    ap.add_argument("--lambda_center", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.25)
    ap.add_argument("--norm", default="per_batch",
                    choices=["per_batch", "source_only", "global", "none"],
                    help="normalization protocol (Dataset B only)")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps")
    ap.add_argument("--save_checkpoint", default=None,
                    help="path for the checkpoint of the first seed")
    ap.add_argument("--out", default=None, help="optional JSON results path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dataset = args.dataset.upper()
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data" / f"DataSet{dataset}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    num_classes = 6 if dataset == "A" else 4

    print(f"UnifiedGas | Dataset {dataset} | batch/board {args.source} -> {args.target}")
    print(f"  data      : {data_dir}")
    print(f"  device    : {device}")
    print(f"  epochs    : {args.epochs}   seeds: {seeds}")

    sx, sy, tx, ty, norm_stats = load_pair(dataset, data_dir, args.source,
                                           args.target, args.norm)
    print(f"  source    : {tuple(sx.shape)}  target: {tuple(tx.shape)}")

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lambda_coral=args.lambda_coral, lambda_center=args.lambda_center,
        warmup_frac=args.warmup_frac, num_classes=num_classes, dataset=dataset,
    )

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[seed {seed}]")
        src_ld, tgt_tr_ld, tgt_ev_ld = make_loaders(sx, sy, tx, ty, args.batch_size)
        trainer = UnifiedGasTrainer(cfg, device=device)
        t0 = time.time()
        res = trainer.fit(src_ld, tgt_tr_ld, tgt_ev_ld, seed=seed, verbose=not args.quiet)
        res["wall_time_s"] = round(time.time() - t0, 1)
        results.append(res)
        print(f"  best={res['best_acc'] * 100:.2f}% @ep{res['best_epoch']}  "
              f"final={res['final_acc'] * 100:.2f}%  ({res['wall_time_s']}s)")
        if args.save_checkpoint and i == 0:
            p = trainer.save_checkpoint(args.save_checkpoint, extra={
                "dataset": dataset, "source": args.source, "target": args.target,
                "seed": seed, "best_acc": res["best_acc"],
                "num_classes": num_classes,
                # Recorded so that predict.py applies the same input protocol;
                # without it a Dataset B checkpoint sees unnormalized inputs.
                # The fitted statistics travel with it, so source_only/global
                # reapply the training transform instead of refitting on the
                # data they are shown.
                "norm": args.norm,
                "norm_stats": (None if norm_stats is None else
                               (norm_stats[0].tolist(), norm_stats[1].tolist())),
                "best_epoch": res["best_epoch"],
            })
            print(f"  [saved] {p}")

    best = [r["best_acc"] for r in results]
    final = [r["final_acc"] for r in results]
    summary = {
        "dataset": dataset, "source": args.source, "target": args.target,
        "epochs": args.epochs, "seeds": seeds,
        "best_acc_mean": float(np.mean(best)), "best_acc_std": float(np.std(best)),
        "final_acc_mean": float(np.mean(final)),
        "per_seed": results,
    }
    print(f"\n{'=' * 60}")
    print(f"best-epoch accuracy : {np.mean(best) * 100:.2f}% +/- {np.std(best) * 100:.2f}")
    print(f"final-epoch accuracy: {np.mean(final) * 100:.2f}%")
    print(f"{'=' * 60}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
