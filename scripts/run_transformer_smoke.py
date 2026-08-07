#!/usr/bin/env python3
"""Run the capacity-matched Transformer smoke experiment on Dataset A.

The smoke stage evaluates one fixed TMSCA-size Transformer configuration on
three representative fixed-source transfers (Batch 1 -> 6, 8, and 9). It runs
both source-only classification and the full UnifiedGas objective with the same
encoder, optimizer, epoch budget, and retrospective reporting rule.

Example
-------
    python scripts/run_transformer_smoke.py \
        --data_dir data/DataSetA --targets 6,8,9 --epochs 400 --seed 42 \
        --device cuda --out results/transformer_smoke.json

This script does not tune a configuration per target. The same arguments are
applied to every task, and the JSON output records best-epoch, final-epoch,
parameter-count, and latency measurements for the go/no-go decision.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unifiedgas import (  # noqa: E402
    GasClassifier,
    TrainConfig,
    UnifiedGasTrainer,
    load_dataset_a_batch,
    make_loaders,
    multi_level_cls_loss,
    set_seed,
)


class SourceOnlyTrainer(UnifiedGasTrainer):
    """Use the shared Transformer and heads without target-domain losses."""

    def fit(self, src_loader, tgt_train_loader, tgt_eval_loader,
            seed: int = 42, verbose: bool = True) -> dict:
        set_seed(seed)
        cfg = self.cfg
        self.model = self.build_model()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.eta_min,
        )
        best_acc, best_epoch, final_acc = 0.0, -1, 0.0
        self.best_state = None
        self.history = []

        for epoch in range(cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_steps = 0
            for (source_x, source_y), _ in zip(src_loader, cycle(tgt_train_loader)):
                source_x = source_x.to(self.device)
                source_y = source_y.to(self.device)
                (out1, out2, fused), center = self.model(source_x, source_y)
                loss = multi_level_cls_loss(
                    out1,
                    out2,
                    fused,
                    source_y,
                    cfg.cls_weights,
                )
                loss = loss + cfg.lambda_center * center
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_steps += 1

            scheduler.step()
            accuracy, _, _ = self.evaluate(tgt_eval_loader)
            final_acc = accuracy
            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch
                self.best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            self.history.append({
                "epoch": epoch,
                "loss": epoch_loss / max(n_steps, 1),
                "target_acc": accuracy,
                "lr": scheduler.get_last_lr()[0],
            })
            if verbose and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
                print(
                    f"  epoch {epoch:4d}  "
                    f"loss={epoch_loss / max(n_steps, 1):.4f}  "
                    f"target_acc={accuracy * 100:.2f}%",
                    flush=True,
                )

        return {
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "final_acc": final_acc,
            "seed": seed,
        }


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def benchmark_latency(model: nn.Module, device: torch.device,
                      batch_size: int = 256, repetitions: int = 100) -> dict:
    """Measure batched inference latency on synthetic Dataset-A inputs."""
    model.eval()
    sample = torch.randn(batch_size, 1, 16, 8, device=device)
    warmup_steps = min(20, repetitions)
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "repetitions": repetitions,
        "batch_latency_ms": total_seconds * 1000.0 / repetitions,
        "per_sample_latency_ms": (
            total_seconds * 1000.0 / repetitions / batch_size
        ),
    }


def run_task(method: str, source_x: torch.Tensor, source_y: torch.Tensor,
             target_x: torch.Tensor, target_y: torch.Tensor,
             config: TrainConfig, device: torch.device, seed: int,
             verbose: bool) -> dict:
    """Run one method on one source-target task."""
    source_loader, target_train_loader, target_eval_loader = make_loaders(
        source_x,
        source_y,
        target_x,
        target_y,
        config.batch_size,
    )
    trainer_type = SourceOnlyTrainer if method == "source_only" else UnifiedGasTrainer
    trainer = trainer_type(config, device=device)
    started_at = time.perf_counter()
    result = trainer.fit(
        source_loader,
        target_train_loader,
        target_eval_loader,
        seed=seed,
        verbose=verbose,
    )
    result["wall_time_s"] = time.perf_counter() - started_at
    return result


def run_task_process(method: str, source: int, target: int,
                     data_dir: str, config_dict: dict, seed: int,
                     gpu_index: int, verbose: bool) -> tuple[str, dict]:
    """Load one task and run it on one dedicated GPU process."""
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    source_x, source_y = load_dataset_a_batch(
        Path(data_dir) / f"batch{source}.dat"
    )
    target_x, target_y = load_dataset_a_batch(
        Path(data_dir) / f"batch{target}.dat"
    )
    config = TrainConfig(**config_dict)
    result = run_task(
        method,
        source_x,
        source_y,
        target_x,
        target_y,
        config,
        device,
        seed,
        verbose=verbose,
    )
    return f"B{source}->B{target}|{method}", result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capacity-matched Transformer smoke experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="data/DataSetA")
    parser.add_argument("--targets", default="6,8,9")
    parser.add_argument("--source", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_frac", type=float, default=0.25)
    parser.add_argument("--lambda_coral", type=float, default=1.0)
    parser.add_argument("--lambda_center", type=float, default=0.1)
    parser.add_argument("--methods", default="source_only,unifiedgas")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--parallel_gpus",
        action="store_true",
        help="run each method/target pair in a dedicated CUDA process",
    )
    parser.add_argument(
        "--out",
        default="results/transformer_experiment/smoke.json",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    targets = [int(target) for target in args.targets.split(",")]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid_methods = sorted(set(methods) - {"source_only", "unifiedgas"})
    if invalid_methods:
        parser.error(f"unsupported methods: {', '.join(invalid_methods)}")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        lambda_coral=args.lambda_coral,
        lambda_center=args.lambda_center,
        num_classes=6,
        dataset="A",
        backbone="transformer",
    )
    probe = GasClassifier(
        num_classes=6,
        dataset="A",
        dropout_rate=config.dropout,
        backbone="transformer",
    ).to(device)
    parameter_count = count_parameters(probe)
    latency = benchmark_latency(probe, device)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("UnifiedGas Transformer smoke")
    print(f"  data       : {data_dir}")
    print(f"  source     : Batch {args.source}")
    print(f"  targets    : {targets}")
    print(f"  methods    : {methods}")
    print(f"  seed       : {args.seed}")
    print(f"  device     : {device}")
    print(f"  parameters : {parameter_count:,} ({parameter_count / 1e6:.3f} M)")
    print(f"  latency    : {latency['per_sample_latency_ms']:.4f} ms/sample")

    results: dict = {
        "experiment": "tmsca_size_transformer_smoke",
        "architecture": {
            "tokens": 16,
            "features_per_token": 8,
            "d_model": 128,
            "num_layers": 6,
            "num_heads": 4,
            "dim_feedforward": 256,
            "parameter_count": parameter_count,
        },
        "latency": latency,
        "config": asdict(config),
        "source": args.source,
        "targets": targets,
        "seed": args.seed,
        "methods": methods,
        "results": {},
    }

    jobs = [(target, method) for target in targets for method in methods]
    if args.parallel_gpus:
        if device.type != "cuda":
            parser.error("--parallel_gpus requires CUDA")
        available_gpus = torch.cuda.device_count()
        if available_gpus < len(jobs):
            parser.error(
                f"--parallel_gpus needs {len(jobs)} GPUs, but only "
                f"{available_gpus} are visible"
            )
        print(f"  parallel   : {len(jobs)} tasks on {available_gpus} visible GPUs")
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    run_task_process,
                    method,
                    args.source,
                    target,
                    str(data_dir),
                    asdict(config),
                    args.seed,
                    gpu_index,
                    not args.quiet,
                ): (target, method)
                for gpu_index, (target, method) in enumerate(jobs)
            }
            for future in as_completed(futures):
                key, result = future.result()
                results["results"][key] = result
                print(
                    f"[{key}] best={result['best_acc'] * 100:.2f}% "
                    f"@ep{result['best_epoch']}  "
                    f"final={result['final_acc'] * 100:.2f}%  "
                    f"wall={result['wall_time_s']:.1f}s",
                    flush=True,
                )
    else:
        source_x, source_y = load_dataset_a_batch(
            data_dir / f"batch{args.source}.dat"
        )
        for target, method in jobs:
            target_x, target_y = load_dataset_a_batch(
                data_dir / f"batch{target}.dat"
            )
            print(f"\n[{method}] Batch {args.source} -> Batch {target}")
            result = run_task(
                method,
                source_x,
                source_y,
                target_x,
                target_y,
                config,
                device,
                args.seed,
                verbose=not args.quiet,
            )
            key = f"B{args.source}->B{target}|{method}"
            results["results"][key] = result
            print(
                f"  best={result['best_acc'] * 100:.2f}% "
                f"@ep{result['best_epoch']}  "
                f"final={result['final_acc'] * 100:.2f}%  "
                f"wall={result['wall_time_s']:.1f}s"
            )

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=2)
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
