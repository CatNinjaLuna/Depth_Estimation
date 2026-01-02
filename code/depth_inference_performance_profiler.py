"""
depth_inference_performance_profiler.py

Computational performance profiling for depth estimation models, aligned
with the multi-dimensional metric taxonomy described in the manuscript:

    - Inference latency (single-frame and batched)
    - Throughput (frames per second)
    - GPU memory usage
    - Approximate GPU utilization and power draw (via nvidia-smi if available)

This script is designed to be run *after* models have been trained, but it
can also operate with randomly initialized networks to compare relative
complexity trends across architectures and resolutions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from depth_model_zoo_and_factories import DepthModelConfig, build_depth_model


@dataclass
class PerformanceConfig:
    arch: str
    input_height: int = 384
    input_width: int = 384
    batch_sizes: Tuple[int, ...] = (1, 4, 8)
    device: str = "cuda"
    n_warmup: int = 10
    n_iters: int = 50
    measure_gpu_stats: bool = True
    output_dir: str = "depth_perf_profiles"


def _generate_dummy_batch(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.rand(batch_size, 3, height, width, device=device)
    return x


def _capture_nvidia_smi_stats() -> Dict[str, float]:
    """
    Query GPU utilization and power draw via nvidia-smi.

    This function is best-effort: if nvidia-smi is not available, it will
    return an empty dict instead of raising an error.
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        line = result.strip().splitlines()[0]
        util_gpu_str, util_mem_str, power_str = [x.strip() for x in line.split(",")]
        return {
            "gpu_util_percent": float(util_gpu_str),
            "mem_util_percent": float(util_mem_str),
            "power_watts": float(power_str),
        }
    except Exception:
        return {}


def profile_single_configuration(
    cfg: PerformanceConfig,
    batch_size: int,
) -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model_cfg = DepthModelConfig(
        arch=cfg.arch,
        base_channels=64 if cfg.arch != "lightweight_cnn" else 32,
        transformer_depth=4 if cfg.arch in {"transformer", "hybrid_cnn_transformer"} else 0,
        num_heads=4,
    )
    model = build_depth_model(model_cfg).to(device)
    model.eval()

    x = _generate_dummy_batch(
        batch_size=batch_size,
        height=cfg.input_height,
        width=cfg.input_width,
        device=device,
    )
    depth_gt = torch.rand(batch_size, 1, cfg.input_height, cfg.input_width, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up iterations (excluded from timing statistics)
    for _ in range(cfg.n_warmup):
        with torch.no_grad():
            y = model(x)
            loss = torch.mean((y - depth_gt) ** 2)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed iterations
    times_ms: List[float] = []
    for _ in range(cfg.n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            y = model(x)
            loss = torch.mean((y - depth_gt) ** 2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0
        times_ms.append(elapsed_ms)

    mean_latency = float(np.mean(times_ms))
    std_latency = float(np.std(times_ms))
    fps = float(batch_size * 1000.0 / mean_latency)

    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = float(peak_mem_bytes / (1024 ** 2))
    else:
        peak_mem_mb = 0.0

    stats = {
        "arch": cfg.arch,
        "batch_size": batch_size,
        "input_height": cfg.input_height,
        "input_width": cfg.input_width,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "throughput_fps": fps,
        "peak_mem_mb": peak_mem_mb,
    }

    if cfg.measure_gpu_stats and device.type == "cuda":
        stats.update(_capture_nvidia_smi_stats())

    return stats


def run_profiling(cfg: PerformanceConfig) -> List[Dict[str, float]]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    rows: List[Dict[str, float]] = []

    for bs in cfg.batch_sizes:
        print(
            f"[Profile] arch={cfg.arch}, input={cfg.input_height}x{cfg.input_width}, "
            f"batch_size={bs}"
        )
        stats = profile_single_configuration(cfg, batch_size=bs)
        rows.append(stats)
        print(
            f"  mean_latency={stats['mean_latency_ms']:.2f} ms | "
            f"throughput={stats['throughput_fps']:.2f} FPS | "
            f"peak_mem={stats['peak_mem_mb']:.1f} MB"
        )

    out_json = os.path.join(
        cfg.output_dir,
        f"perf_profile_{cfg.arch}_{cfg.input_height}x{cfg.input_width}.json",
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "results": rows}, f, indent=2)

    try:
        import pandas as pd  # type: ignore

        out_csv = os.path.join(
            cfg.output_dir,
            f"perf_profile_{cfg.arch}_{cfg.input_height}x{cfg.input_width}.csv",
        )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    except Exception:
        # Pandas is optional; if not available, users still have the JSON file.
        pass

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference performance profiler for depth estimation models."
    )
    parser.add_argument("--arch", type=str, default="cnn_baseline")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output-dir", type=str, default="depth_perf_profiles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PerformanceConfig(
        arch=args.arch,
        input_height=args.height,
        input_width=args.width,
        batch_sizes=tuple(args.batch_sizes),
        output_dir=args.output_dir,
    )
    run_profiling(cfg)


if __name__ == "__main__":
    main()
