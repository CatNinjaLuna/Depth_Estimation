"""
depth_results_aggregation_and_latex_export.py

Result aggregation and LaTeX table generator for the depth estimation
benchmarking study.

This utility collects outputs from multiple experimental scripts,
including:

    - Training & evaluation summaries
    - Inference performance profiles
    - Robustness evaluation under environmental conditions
    - Hyper-parameter & resolution ablation

and converts them into consolidated CSV files and LaTeX tables that
can be dropped directly into the manuscript.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _load_jsons(pattern: str) -> List[Dict]:
    paths = sorted(glob.glob(pattern))
    records: List[Dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj["_source_path"] = p
            records.append(obj)
    return records


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def aggregate_training_summaries(summary_dir: str) -> pd.DataFrame:
    """
    Aggregate summary_*.json files produced by depth_training_and_evaluation_pipeline.py
    into a single DataFrame.
    """
    records = _load_jsons(os.path.join(summary_dir, "summary_*.json"))
    rows: List[Dict] = []
    for rec in records:
        cfg = rec.get("config", {})
        test_stats = rec.get("test_stats", {})
        row = {
            "arch": cfg.get("arch"),
            "resize_h": cfg.get("resize_h"),
            "resize_w": cfg.get("resize_w"),
            "max_epochs": cfg.get("max_epochs"),
            "base_lr": cfg.get("base_lr"),
            "weight_decay": cfg.get("weight_decay"),
            "best_val_absrel": rec.get("best_val_absrel"),
            "test_AbsRel": test_stats.get("AbsRel"),
            "test_RMSE": test_stats.get("RMSE"),
            "test_delta<1.25": test_stats.get("delta<1.25"),
            "test_delta<1.5625": test_stats.get("delta<1.5625"),
            "test_delta<1.953125": test_stats.get("delta<1.953125"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_perf_profiles(perf_dir: str) -> pd.DataFrame:
    """
    Aggregate perf_profile_*.json outputs from depth_inference_performance_profiler.py.
    """
    records = _load_jsons(os.path.join(perf_dir, "perf_profile_*.json"))
    rows: List[Dict] = []
    for rec in records:
        cfg = rec.get("config", {})
        for r in rec.get("results", []):
            row = {
                "arch": cfg.get("arch"),
                "input_height": cfg.get("input_height"),
                "input_width": cfg.get("input_width"),
                **r,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_robustness(robust_csv: str) -> pd.DataFrame:
    if not os.path.exists(robust_csv):
        return pd.DataFrame()
    return pd.read_csv(robust_csv)


def aggregate_ablation(ablation_dir: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(ablation_dir, "depth_ablation_results*.csv")))
    if not csv_paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def df_to_latex(
    df: pd.DataFrame,
    cols: List[str],
    caption: str,
    label: str,
    float_format: str = "%.3f",
) -> str:
    sub = df[cols].copy()
    latex = sub.to_latex(
        index=False,
        float_format=lambda x: float_format % x,
        escape=True,
        caption=caption,
        label=label,
    )
    return latex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate depth benchmark results and export LaTeX tables."
    )
    parser.add_argument("--summary-dir", type=str, default="depth_benchmark_runs")
    parser.add_argument("--perf-dir", type=str, default="depth_perf_profiles")
    parser.add_argument(
        "--robustness-csv", type=str, default="depth_robustness_environmental.csv"
    )
    parser.add_argument("--ablation-dir", type=str, default="depth_ablation_runs")
    parser.add_argument("--output-dir", type=str, default="depth_aggregated_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Main accuracy results
    df_train = aggregate_training_summaries(args.summary_dir)
    if not df_train.empty:
        df_train.to_csv(
            os.path.join(args.output_dir, "depth_training_summary_agg.csv"),
            index=False,
        )
        latex_main = df_to_latex(
            df_train,
            cols=[
                "arch",
                "resize_h",
                "resize_w",
                "test_AbsRel",
                "test_RMSE",
                "test_delta<1.25",
            ],
            caption=(
                "Main depth estimation results in unstructured environments. "
                "Lower AbsRel/RMSE is better; higher $\\delta<1.25$ is better."
            ),
            label="tab:depth_main_results",
        )
        with open(
            os.path.join(args.output_dir, "table_depth_main_results.tex"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(latex_main)

    # 2) Inference performance
    df_perf = aggregate_perf_profiles(args.perf_dir)
    if not df_perf.empty:
        df_perf.to_csv(
            os.path.join(args.output_dir, "depth_perf_profiles_agg.csv"),
            index=False,
        )
        latex_perf = df_to_latex(
            df_perf,
            cols=[
                "arch",
                "batch_size",
                "input_height",
                "input_width",
                "mean_latency_ms",
                "throughput_fps",
                "peak_mem_mb",
            ],
            caption=(
                "Inference performance across architectures and batch sizes. "
                "Latency is measured in milliseconds; throughput is frames per second."
            ),
            label="tab:depth_perf",
        )
        with open(
            os.path.join(args.output_dir, "table_depth_perf.tex"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(latex_perf)

    # 3) Robustness table
    df_rob = aggregate_robustness(args.robustness_csv)
    if not df_rob.empty:
        df_rob.to_csv(
            os.path.join(args.output_dir, "depth_robustness_agg.csv"),
            index=False,
        )
        latex_rob = df_to_latex(
            df_rob,
            cols=["arch", "corruption", "severity", "AbsRel", "RMSE"],
            caption=(
                "Robustness of a reference depth model under simulated "
                "environmental degradations."
            ),
            label="tab:depth_robustness",
        )
        with open(
            os.path.join(args.output_dir, "table_depth_robustness.tex"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(latex_rob)

    # 4) Ablation table
    df_ablation = aggregate_ablation(args.ablation_dir)
    if not df_ablation.empty:
        df_ablation.to_csv(
            os.path.join(args.output_dir, "depth_ablation_agg.csv"),
            index=False,
        )
        latex_ablation = df_to_latex(
            df_ablation,
            cols=[
                "resolution_h",
                "resolution_w",
                "lr",
                "weight_decay",
                "test_AbsRel",
                "test_RMSE",
            ],
            caption=(
                "Ablation over input resolution and optimization hyper-parameters. "
                "Metrics are reported on the held-out test split."
            ),
            label="tab:depth_ablation",
        )
        with open(
            os.path.join(args.output_dir, "table_depth_ablation.tex"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(latex_ablation)


if __name__ == "__main__":
    main()
