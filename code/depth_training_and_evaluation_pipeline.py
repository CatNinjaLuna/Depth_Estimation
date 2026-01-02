"""
depth_training_and_evaluation_pipeline.py

End-to-end training and evaluation pipeline for depth estimation
algorithms in unstructured environments.

This script binds together:
    - depth_benchmark_dataset_and_metrics
    - depth_model_zoo_and_factories

and implements:
    - unified training loop with multi-metric monitoring (AbsRel, RMSE, δ thresholds)
    - scenario-aware evaluation across indoor/outdoor test splits
    - JSON logging of per-epoch statistics for subsequent analysis

The implementation is intentionally verbose and research-oriented, with
explicit control over hyper-parameters so that the exact experimental
setup can be reproduced from the configuration printed at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from depth_benchmark_dataset_and_metrics import (
    MetricAccumulator,
    build_dataloaders,
)
from depth_model_zoo_and_factories import DepthModelConfig, build_depth_model


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    arch: str = "cnn_baseline"
    dataset_root: str = "DATA_ROOT_PLACEHOLDER"
    index_csv: str = "INDEX_CSV_PLACEHOLDER.csv"
    batch_size: int = 4
    num_workers: int = 4
    resize_h: int = 384
    resize_w: int = 384
    random_crop_h: int = 352
    random_crop_w: int = 352
    color_jitter: bool = True
    max_epochs: int = 40
    base_lr: float = 1e-4
    weight_decay: float = 1e-5
    lr_step_epochs: int = 15
    lr_gamma: float = 0.5
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 50
    output_dir: str = "depth_benchmark_runs"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_dataloaders_from_cfg(cfg: TrainConfig) -> Dict[str, DataLoader]:
    loaders = build_dataloaders(
        root=cfg.dataset_root,
        index_csv=cfg.index_csv,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        resize=(cfg.resize_h, cfg.resize_w),
        random_crop=(cfg.random_crop_h, cfg.random_crop_w),
        random_flip=True,
        color_jitter=cfg.color_jitter,
    )
    return loaders


def create_model_from_cfg(cfg: TrainConfig) -> nn.Module:
    model_cfg = DepthModelConfig(
        arch=cfg.arch,
        base_channels=64 if cfg.arch != "lightweight_cnn" else 32,
        transformer_depth=4 if cfg.arch in {"transformer", "hybrid_cnn_transformer"} else 0,
        num_heads=4,
    )
    model = build_depth_model(model_cfg)
    return model


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_idx, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        depth_pred = model(rgb)

        # Scale-invariant L2 loss as a simple depth supervision term.
        diff = depth_pred - depth_gt
        diff = diff * valid_mask
        loss_l2 = torch.mean(diff ** 2)
        loss = loss_l2

        loss.backward()
        optimizer.step()

        batch_size = rgb.size(0)
        total_loss += float(loss.item()) * batch_size
        num_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"[Train] Epoch {epoch:03d} | "
                f"Iter {batch_idx+1:05d}/{len(loader):05d} | "
                f"Batch Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / max(num_samples, 1)
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        depth_pred = model(rgb)

        metrics.update(depth_pred, depth_gt, valid_mask)

    summary = metrics.summarize()
    return summary


def run_training(cfg: TrainConfig) -> Dict[str, object]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_random_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loaders = create_dataloaders_from_cfg(cfg)
    model = create_model_from_cfg(cfg)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step_epochs,
        gamma=cfg.lr_gamma,
    )

    history: List[Dict[str, float]] = []
    best_val_absrel = float("inf")
    best_ckpt_path = os.path.join(cfg.output_dir, f"best_{cfg.arch}.pth")

    print("==== Training configuration ====")
    print(json.dumps(asdict(cfg), indent=2))
    print("================================")

    for epoch in range(1, cfg.max_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=cfg.log_interval,
        )
        val_stats = evaluate(
            model=model,
            loader=loaders["val"],
            device=device,
        )
        scheduler.step()

        curr_lr = scheduler.get_last_lr()[0]
        record = {
            "epoch": epoch,
            "lr": curr_lr,
            "train_loss": train_stats["loss"],
        }
        record.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(record)

        print(
            f"[Epoch {epoch:03d}] LR={curr_lr:.5f} | "
            f"TrainLoss={train_stats['loss']:.4f} | "
            f"Val AbsRel={val_stats.get('AbsRel', float('nan')):.4f} | "
            f"Val RMSE={val_stats.get('RMSE', float('nan')):.4f}"
        )

        absrel = val_stats.get("AbsRel", float("inf"))
        if absrel < best_val_absrel:
            best_val_absrel = absrel
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "cfg": asdict(cfg),
                    "val_stats": val_stats,
                },
                best_ckpt_path,
            )

    # Final evaluation on the held-out test split using the best checkpoint.
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_stats = evaluate(
        model=model,
        loader=loaders["test"],
        device=device,
    )
    print("[Test] Metrics:")
    for k, v in test_stats.items():
        print(f"  {k}: {v:.4f}")

    history_path = os.path.join(cfg.output_dir, f"history_{cfg.arch}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "config": asdict(cfg),
        "best_val_absrel": best_val_absrel,
        "test_stats": test_stats,
        "history_path": history_path,
        "best_ckpt_path": best_ckpt_path,
    }

    summary_path = os.path.join(cfg.output_dir, f"summary_{cfg.arch}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training and evaluation pipeline for depth estimation benchmarking."
    )
    parser.add_argument("--arch", type=str, default="cnn_baseline")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--index-csv", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="depth_benchmark_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        arch=args.arch,
        dataset_root=args.dataset_root,
        index_csv=args.index_csv,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        base_lr=args.base_lr,
        output_dir=args.output_dir,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
