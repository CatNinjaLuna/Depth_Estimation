"""
depth_hyperparameter_and_resolution_ablation.py

Hyper-parameter and input resolution ablation driver for depth
estimation benchmarking in unstructured environments.

This script programmatically creates multiple TrainConfig instances
with different learning rates, weight decays, and input resolutions,
invokes the main training pipeline, and aggregates the resulting
metrics into a single CSV file.

The output can be used directly to populate ablation tables (e.g.,
the effect of resolution and optimization settings on AbsRel and RMSE).
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from depth_training_and_evaluation_pipeline import TrainConfig, run_training


@dataclass
class AblationSearchSpace:
    learning_rates: Iterable[float]
    weight_decays: Iterable[float]
    resolutions: Iterable[Tuple[int, int]]  # (H, W)
    seeds: Iterable[int]


@dataclass
class AblationConfig:
    arch: str
    dataset_root: str
    index_csv: str
    base_batch_size: int = 4
    max_epochs: int = 30
    output_dir: str = "depth_ablation_runs"
    results_csv: str = "depth_ablation_results.csv"


def run_ablation(
    cfg: AblationConfig,
    space: AblationSearchSpace,
) -> List[Dict[str, object]]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for (h, w) in space.resolutions:
        for lr in space.learning_rates:
            for wd in space.weight_decays:
                for seed in space.seeds:
                    train_cfg = TrainConfig(
                        arch=cfg.arch,
                        dataset_root=cfg.dataset_root,
                        index_csv=cfg.index_csv,
                        batch_size=cfg.base_batch_size,
                        resize_h=h,
                        resize_w=w,
                        random_crop_h=h - 16 if h > 32 else h,
                        random_crop_w=w - 16 if w > 32 else w,
                        max_epochs=cfg.max_epochs,
                        base_lr=lr,
                        weight_decay=wd,
                        seed=seed,
                        output_dir=cfg.output_dir,
                    )
                    print(
                        f"[Ablation] arch={cfg.arch}, res={h}x{w}, "
                        f"lr={lr:.1e}, wd={wd:.1e}, seed={seed}"
                    )
                    summary = run_training(train_cfg)
                    test_stats = summary["test_stats"]
                    row = {
                        "arch": cfg.arch,
                        "resolution_h": h,
                        "resolution_w": w,
                        "lr": lr,
                        "weight_decay": wd,
                        "seed": seed,
                        "best_val_absrel": summary["best_val_absrel"],
                        "test_AbsRel": test_stats.get("AbsRel"),
                        "test_RMSE": test_stats.get("RMSE"),
                        "test_delta<1.25": test_stats.get("delta<1.25"),
                        "test_delta<1.5625": test_stats.get("delta<1.5625"),
                        "test_delta<1.953125": test_stats.get("delta<1.953125"),
                    }
                    rows.append(row)

    # Write aggregated CSV
    out_csv_path = os.path.join(cfg.output_dir, cfg.results_csv)
    fieldnames = [
        "arch",
        "resolution_h",
        "resolution_w",
        "lr",
        "weight_decay",
        "seed",
        "best_val_absrel",
        "test_AbsRel",
        "test_RMSE",
        "test_delta<1.25",
        "test_delta<1.5625",
        "test_delta<1.953125",
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyper-parameter and resolution ablation for depth models."
    )
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--index-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="depth_ablation_runs")
    parser.add_argument("--max-epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AblationConfig(
        arch=args.arch,
        dataset_root=args.dataset_root,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
    )
    space = AblationSearchSpace(
        learning_rates=[5e-5, 1e-4, 2e-4],
        weight_decays=[1e-6, 1e-5, 1e-4],
        resolutions=[(256, 256), (384, 384), (480, 640)],
        seeds=[42, 123],
    )
    run_ablation(cfg, space)


if __name__ == "__main__":
    main()
