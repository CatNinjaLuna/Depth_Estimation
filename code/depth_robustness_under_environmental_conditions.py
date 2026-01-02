"""
depth_robustness_under_environmental_conditions.py

Robustness benchmarking for depth estimation models under simulated
environmental degradations in unstructured scenes.

This script evaluates a trained depth model on the held-out test split
while injecting controlled input perturbations that emulate realistic
operational challenges discussed in the paper, including:

    - Illumination changes (brightness / contrast)
    - Image noise (Gaussian)
    - Blur (Gaussian blur)
    - Partial occlusion (random rectangular masks)
    - Resolution degradation (downsample & upsample)

For each corruption type and severity level, the script computes
dataset-level AbsRel, RMSE and δ-threshold accuracies, and stores the
results in a CSV table that can be directly converted into a robustness
figure or table in the experimental section.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depth_benchmark_dataset_and_metrics import (
    MetricAccumulator,
    build_dataloaders,
)
from depth_model_zoo_and_factories import DepthModelConfig, build_depth_model


@dataclass
class RobustnessConfig:
    arch: str
    checkpoint_path: str
    dataset_root: str
    index_csv: str
    batch_size: int = 4
    num_workers: int = 4
    resize_h: int = 384
    resize_w: int = 384
    device: str = "cuda"
    severities: Tuple[int, ...] = (1, 2, 3, 4, 5)
    output_csv: str = "depth_robustness_environmental.csv"


# ---------------------------------------------------------------------------
# Corruption operators
# ---------------------------------------------------------------------------


def _apply_brightness_contrast(
    rgb: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    # rgb: (B, 3, H, W), normalized to roughly [-1, 1] or [0, 1].
    factor_b = 0.1 * severity  # brightness
    factor_c = 0.1 * severity  # contrast
    mean = rgb.mean(dim=(2, 3), keepdim=True)
    rgb_adj = (rgb - mean) * (1.0 + factor_c) + mean + factor_b
    return rgb_adj


def _apply_gaussian_noise(
    rgb: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    sigma = 0.03 * severity
    noise = torch.randn_like(rgb) * sigma
    return rgb + noise


def _apply_gaussian_blur(
    rgb: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    # Use a simple separable Gaussian via conv.
    kernel_sizes = {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}
    k = kernel_sizes.get(severity, 11)
    sigma = 0.5 * severity

    radius = k // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=rgb.device)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k, k)
    kernel = kernel.repeat(rgb.shape[1], 1, 1, 1)  # channel-wise

    padding = radius
    rgb_pad = F.pad(rgb, (padding, padding, padding, padding), mode="reflect")
    rgb_blur = F.conv2d(rgb_pad, kernel, groups=rgb.shape[1])
    return rgb_blur


def _apply_random_occlusion(
    rgb: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    # Draw random rectangles covering a fraction of the image area.
    b, c, h, w = rgb.shape
    frac = 0.08 * severity  # target area fraction
    area = h * w
    occl_area = frac * area
    side = int((occl_area) ** 0.5)
    side = max(8, min(side, min(h, w) // 2))

    out = rgb.clone()
    for i in range(b):
        top = np.random.randint(0, max(1, h - side))
        left = np.random.randint(0, max(1, w - side))
        out[i, :, top : top + side, left : left + side] = 0.0
    return out


def _apply_resolution_degradation(
    rgb: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    # Downsample to a smaller resolution and upsample back.
    scale = 1.0 / (1.0 + 0.5 * severity)
    b, c, h, w = rgb.shape
    h_low = max(16, int(h * scale))
    w_low = max(16, int(w * scale))
    x_low = F.interpolate(rgb, size=(h_low, w_low), mode="bilinear", align_corners=False)
    x_up = F.interpolate(x_low, size=(h, w), mode="bilinear", align_corners=False)
    return x_up


def apply_corruption(
    rgb: torch.Tensor,
    corruption: str,
    severity: int,
) -> torch.Tensor:
    if corruption == "brightness_contrast":
        return _apply_brightness_contrast(rgb, severity)
    if corruption == "gaussian_noise":
        return _apply_gaussian_noise(rgb, severity)
    if corruption == "gaussian_blur":
        return _apply_gaussian_blur(rgb, severity)
    if corruption == "random_occlusion":
        return _apply_random_occlusion(rgb, severity)
    if corruption == "resolution_degradation":
        return _apply_resolution_degradation(rgb, severity)
    raise ValueError(f"Unknown corruption type: {corruption}")


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def _load_model(cfg: RobustnessConfig) -> torch.nn.Module:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model_cfg = DepthModelConfig(
        arch=cfg.arch,
        base_channels=64 if cfg.arch != "lightweight_cnn" else 32,
        transformer_depth=4 if cfg.arch in {"transformer", "hybrid_cnn_transformer"} else 0,
        num_heads=4,
    )
    model = build_depth_model(model_cfg)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_robustness(cfg: RobustnessConfig) -> List[Dict[str, float]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(
        root=cfg.dataset_root,
        index_csv=cfg.index_csv,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        resize=(cfg.resize_h, cfg.resize_w),
        random_crop=None,
        random_flip=False,
        color_jitter=False,
    )
    test_loader: DataLoader = loaders["test"]
    model = _load_model(cfg)

    corruption_types = [
        "brightness_contrast",
        "gaussian_noise",
        "gaussian_blur",
        "random_occlusion",
        "resolution_degradation",
    ]

    results: List[Dict[str, float]] = []

    for corruption in corruption_types:
        for severity in cfg.severities:
            accumulator = MetricAccumulator()

            for batch in test_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                depth_gt = batch["depth"].to(device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True)

                rgb_corr = apply_corruption(rgb, corruption, severity)
                with torch.no_grad():
                    depth_pred = model(rgb_corr)

                accumulator.update(depth_pred, depth_gt, valid_mask)

            summary = accumulator.summarize()
            row = {
                "arch": cfg.arch,
                "corruption": corruption,
                "severity": severity,
            }
            row.update(summary)
            results.append(row)

            print(
                f"[Robustness] arch={cfg.arch:20s} | "
                f"corruption={corruption:22s} | severity={severity} | "
                f"AbsRel={summary.get('AbsRel', float('nan')):.4f} | "
                f"RMSE={summary.get('RMSE', float('nan')):.4f}"
            )

    os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)
    fieldnames = [
        "arch",
        "corruption",
        "severity",
        "AbsRel",
        "RMSE",
        "delta<1.25",
        "delta<1.5625",
        "delta<1.953125",
    ]
    with open(cfg.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness evaluation for depth models under environmental conditions."
    )
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--index-csv", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--resize-w", type=int, default=384)
    parser.add_argument("--output-csv", type=str, default="depth_robustness_environmental.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RobustnessConfig(
        arch=args.arch,
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        index_csv=args.index_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        output_csv=args.output_csv,
    )
    evaluate_robustness(cfg)


if __name__ == "__main__":
    main()
