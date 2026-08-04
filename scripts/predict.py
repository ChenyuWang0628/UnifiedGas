#!/usr/bin/env python3
"""Run inference with a trained UnifiedGas checkpoint.

Examples
--------
    # Classify a target batch and report accuracy against its labels
    python scripts/predict.py --checkpoint checkpoints/unifiedgas_A_batch1to6.pt \
        --dataset A --input data/DataSetA/batch6.dat

    # Classify an unlabeled feature matrix stored as .npy, writing predictions
    python scripts/predict.py --checkpoint checkpoints/unifiedgas_A_batch1to6.pt \
        --dataset A --input my_samples.npy --out predictions.csv

    # Measure per-sample latency
    python scripts/predict.py --checkpoint checkpoints/unifiedgas_A_batch1to6.pt \
        --dataset A --input data/DataSetA/batch6.dat --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unifiedgas import (  # noqa: E402
    DATASET_A_CLASSES, DATASET_B_CLASSES, UnifiedGasTrainer,
    apply_norm_stats, load_dataset_a_batch, load_dataset_b_board,
)


def load_input(path: Path, dataset: str, norm: str = "per_batch", stats=None):
    """Load a benchmark file or a raw .npy matrix, applying the input protocol.

    Dataset A files are z-scored inside :func:`load_dataset_a_batch`, matching
    the per-batch protocol used during training.  Dataset B stores raw
    resistance means, so the same normalization the model was trained under has
    to be applied here explicitly -- otherwise the network sees inputs on a
    completely different scale and the predictions collapse.  Under the
    ``source_only`` and ``global`` protocols that means reusing the statistics
    fitted during training (``stats``), not refitting on the incoming data.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if dataset == "B":
            arr = apply_norm_stats(arr, stats, mode=norm).astype(np.float32)
        X = torch.from_numpy(arr)
        if dataset == "A" and X.dim() == 2 and X.size(1) == 128:
            X = X.view(-1, 1, 16, 8)
        return X, None
    if suffix == ".dat":
        return load_dataset_a_batch(path)
    if suffix == ".csv":
        X, y = load_dataset_b_board(path)
        Xn = apply_norm_stats(X.numpy(), stats, mode=norm).astype(np.float32)
        return torch.from_numpy(Xn), y
    raise ValueError(f"unsupported input type: {suffix}")


@torch.no_grad()
def benchmark(model, X, device, n_warmup: int = 20, n_iter: int = 200):
    """Report single-sample and batched inference latency."""
    x1 = X[:1].to(device)
    xb = X.to(device)
    for _ in range(n_warmup):
        model(x1)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x1)
    single_ms = (time.perf_counter() - t0) / n_iter * 1000

    for _ in range(5):
        model(xb)
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        model(xb)
    batched_ms = (time.perf_counter() - t0) / reps / X.size(0) * 1000
    return single_ms, batched_ms


def main():
    ap = argparse.ArgumentParser(
        description="UnifiedGas inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True,
                    help=".dat / .csv benchmark file, or .npy feature matrix")
    ap.add_argument("--dataset", default=None, choices=["A", "B"],
                    help="defaults to the value stored in the checkpoint")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write predictions to this CSV")
    ap.add_argument("--norm", default=None,
                    choices=["per_batch", "source_only", "global", "none"],
                    help="input normalization; defaults to the protocol stored "
                         "in the checkpoint (Dataset B only)")
    ap.add_argument("--benchmark", action="store_true", help="measure latency")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer, payload = UnifiedGasTrainer.load_checkpoint(args.checkpoint, device=device)
    model = trainer.model
    dataset = (args.dataset or payload.get("dataset", "A")).upper()
    class_names = DATASET_A_CLASSES if dataset == "A" else DATASET_B_CLASSES

    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint : {args.checkpoint}")
    print(f"  dataset  : {dataset}   parameters: {n_params / 1e6:.3f} M")
    if "best_acc" in payload:
        print(f"  trained  : batch/board {payload.get('source')} -> {payload.get('target')}"
              f"  (best target acc {payload['best_acc'] * 100:.2f}%)")

    ckpt_norm = payload.get("norm", "per_batch")
    norm = args.norm or ckpt_norm
    stats = payload.get("norm_stats")
    if args.norm and args.norm != ckpt_norm:
        # Overriding the protocol invalidates the stored statistics, which were
        # fitted under the checkpoint's own protocol.
        stats = None
    X, y = load_input(Path(args.input), dataset, norm=norm, stats=stats)
    print(f"  input    : {tuple(X.shape)} from {args.input}")
    if dataset == "B":
        src = "checkpoint statistics" if stats is not None else "fitted on this input"
        print(f"  normaliz.: {norm} ({src})")

    with torch.no_grad():
        model.eval()
        _, _, fused = model(X.to(device))
        probs = torch.softmax(fused, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    counts = np.bincount(preds, minlength=len(class_names))
    print("\npredicted class distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name:<14}{counts[i]:>6}")

    if y is not None:
        acc = float((preds == y.numpy()).mean())
        print(f"\naccuracy against the provided labels: {acc * 100:.2f}%")

    if args.benchmark:
        single_ms, batched_ms = benchmark(model, X, device)
        print(f"\nlatency on {device}:")
        print(f"  single sample : {single_ms:.3f} ms")
        print(f"  batched       : {batched_ms:.3f} ms / sample "
              f"(batch size {X.size(0)})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = "index,prediction,label," + ",".join(f"p_{c}" for c in class_names)
        with open(out, "w") as fh:
            fh.write(header + "\n")
            for i, (p, pr) in enumerate(zip(preds, probs)):
                lab = int(y[i]) if y is not None else -1
                fh.write(f"{i},{class_names[p]},{lab}," +
                         ",".join(f"{v:.4f}" for v in pr) + "\n")
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
