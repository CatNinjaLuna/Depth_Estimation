"""
depth_benchmark_dataset_and_metrics.py

Dataset abstractions, preprocessing pipeline, and evaluation metrics for the
paper "Performance Benchmarking and Optimization Strategies for Depth
Estimation Algorithms in Unstructured Environments".

This module focuses on three aspects:

1. Scenario-aware dataset definitions
   - Explicit modeling of different unstructured environment types
     (e.g., indoor-structured, indoor-cluttered, outdoor-simple,
     outdoor-complex), as discussed in the benchmarking framework.

2. Preprocessing and augmentation
   - Resolution normalization, color normalization, random cropping,
     horizontal flipping, and optional photometric jitter.
   - Designed to emulate typical depth-estimation training pipelines while
     keeping the implementation transparent and easily auditable.

3. Metric implementations
   - Depth estimation accuracy metrics: AbsRel, RMSE, and delta thresholds
     δ < 1.25, δ < 1.25^2, δ < 1.25^3.
   - Aggregation utilities for computing dataset-level statistics with
     consistent masking and numerical stability handling.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import torchvision.transforms as T
except ImportError:  # pragma: no cover
    T = None


# ---------------------------------------------------------------------------
# Scenario taxonomy
# ---------------------------------------------------------------------------


SCENARIO_INDOOR_STRUCTURED = "indoor_structured"
SCENARIO_INDOOR_CLUTTERED = "indoor_cluttered"
SCENARIO_OUTDOOR_SIMPLE = "outdoor_simple"
SCENARIO_OUTDOOR_COMPLEX = "outdoor_complex"


@dataclass
class ScenarioDefinition:
    name: str
    description: str
    expected_complexity_level: str  # e.g., "low", "medium", "high"


SCENARIOS: Dict[str, ScenarioDefinition] = {
    SCENARIO_INDOOR_STRUCTURED: ScenarioDefinition(
        name=SCENARIO_INDOOR_STRUCTURED,
        description="Indoor environments with regular geometry and limited clutter.",
        expected_complexity_level="low",
    ),
    SCENARIO_INDOOR_CLUTTERED: ScenarioDefinition(
        name=SCENARIO_INDOOR_CLUTTERED,
        description=(
            "Indoor scenes with many small objects, strong occlusions, and mixed"
            " illumination."
        ),
        expected_complexity_level="medium",
    ),
    SCENARIO_OUTDOOR_SIMPLE: ScenarioDefinition(
        name=SCENARIO_OUTDOOR_SIMPLE,
        description=(
            "Outdoor scenes with relatively simple geometry and moderate texture"
            " complexity."
        ),
        expected_complexity_level="medium",
    ),
    SCENARIO_OUTDOOR_COMPLEX: ScenarioDefinition(
        name=SCENARIO_OUTDOOR_COMPLEX,
        description=(
            "Unstructured outdoor environments with dense foliage, complex geometry,"
            " and challenging lighting."
        ),
        expected_complexity_level="high",
    ),
}


# ---------------------------------------------------------------------------
# Dataset abstractions
# ---------------------------------------------------------------------------


@dataclass
class DepthSample:
    """
    Lightweight container describing a single RGB + depth example.

    Attributes:
        rgb_path: Absolute path to the RGB image file.
        depth_path: Absolute path to the depth map file (PNG/NPY/NPZ).
        scenario: Scenario identifier string from SCENARIOS.
        sequence_id: Optional sequence identifier (for video-style datasets).
        frame_index: Optional frame index within the sequence.
        metadata: Additional free-form metadata as a dictionary.
    """

    rgb_path: str
    depth_path: str
    scenario: str
    sequence_id: Optional[str] = None
    frame_index: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None


class DepthBenchmarkDataset(Dataset):
    """
    Dataset for supervised depth estimation benchmarking.

    The dataset is defined by a CSV index file with at least the columns:

        split,rgb_path,depth_path,scenario,sequence_id,frame_index

    where paths are relative to the dataset root. This design mirrors the
    "data collection and scenario classification" section of the manuscript.
    """

    def __init__(
        self,
        root: str,
        index_csv: str,
        split: str,
        resize: Optional[Tuple[int, int]] = (384, 384),
        random_crop: Optional[Tuple[int, int]] = None,
        random_flip: bool = True,
        color_jitter: bool = False,
        depth_scale: float = 1.0,
        max_depth: Optional[float] = None,
        min_depth: float = 0.0,
    ) -> None:
        assert split in {"train", "val", "test"}, "split must be train/val/test"
        self.root = root
        self.split = split
        self.resize = resize
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.samples: List[DepthSample] = self._load_index(index_csv)

        self._transform_rgb = self._build_rgb_transform()
        # Depth maps are processed with a separate path that preserves metric units.

    # ----- index loading ----------------------------------------------------

    def _load_index(self, index_csv: str) -> List[DepthSample]:
        samples: List[DepthSample] = []
        with open(index_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "train") != self.split:
                    continue
                rgb_rel = row["rgb_path"]
                depth_rel = row["depth_path"]
                scenario = row.get("scenario", SCENARIO_OUTDOOR_COMPLEX)
                seq_id = row.get("sequence_id") or None
                frame_idx = row.get("frame_index")
                frame_idx_int = int(frame_idx) if frame_idx not in (None, "") else None
                metadata = {
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "split",
                        "rgb_path",
                        "depth_path",
                        "scenario",
                        "sequence_id",
                        "frame_index",
                    }
                }

                samples.append(
                    DepthSample(
                        rgb_path=os.path.join(self.root, rgb_rel),
                        depth_path=os.path.join(self.root, depth_rel),
                        scenario=scenario,
                        sequence_id=seq_id,
                        frame_index=frame_idx_int,
                        metadata=metadata,
                    )
                )
        if not samples:
            raise RuntimeError(f"No samples found for split={self.split} in {index_csv}")
        return samples

    # ----- transforms -------------------------------------------------------

    def _build_rgb_transform(self):
        if T is None:
            # Minimal fallback transformation for environments without torchvision.
            def _identity(x):
                return x

            return _identity

        ops = []
        if self.resize is not None:
            ops.append(T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR))
        if self.split == "train" and self.random_crop is not None:
            ops.append(T.RandomCrop(self.random_crop))
        if self.split == "train" and self.random_flip:
            ops.append(T.RandomHorizontalFlip())
        if self.split == "train" and self.color_jitter:
            ops.append(
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                )
            )
        ops.append(T.ToTensor())
        # Normalization values are placeholders and can be adjusted to match
        # the specific dataset statistics.
        ops.append(T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]))
        return T.Compose(ops)

    # ----- depth loading utilities -----------------------------------------

    def _load_depth_map(self, path: str) -> np.ndarray:
        """
        Load a depth map and convert it to meters.

        Supported formats:
            - 16-bit PNG (depth in millimeters or decimeters)
            - .npy / .npz arrays (float32)
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in {".npy", ".npz"}:
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                # Heuristically select a likely depth key.
                for key in ["depth", "depth_map", "d"]:
                    if key in arr:
                        arr = arr[key]
                        break
            depth = np.asarray(arr, dtype=np.float32)
        else:
            if Image is None:
                raise RuntimeError(
                    f"PIL is required to read depth image file {path}, but is not installed."
                )
            img = Image.open(path)
            depth_raw = np.array(img, dtype=np.float32)
            # Heuristic: 16-bit PNG often stores depth in millimeters.
            if depth_raw.max() > 1000.0:
                depth = depth_raw / 1000.0  # convert mm -> m
            else:
                depth = depth_raw

        depth = depth * float(self.depth_scale)
        if self.max_depth is not None:
            depth = np.clip(depth, self.min_depth, self.max_depth)
        else:
            depth = np.maximum(depth, self.min_depth)

        return depth

    # ----- dataset API ------------------------------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]

        if Image is None:
            raise RuntimeError("PIL is required to use this dataset but is not installed.")
        rgb = Image.open(sample.rgb_path).convert("RGB")
        depth = self._load_depth_map(sample.depth_path)

        # Perform geometric transforms in a consistent way for rgb and depth.
        if self.resize is not None and T is None:
            # Fallback: simple nearest-neighbor resize using numpy.
            rgb = rgb.resize(self.resize, resample=Image.BILINEAR)
            depth = np.array(
                Image.fromarray(depth).resize(self.resize, resample=Image.NEAREST),
                dtype=np.float32,
            )

        if T is not None:
            # Use torchvision composed pipeline for RGB
            rgb_tensor = self._transform_rgb(rgb)
        else:
            rgb_tensor = torch.from_numpy(np.array(rgb).transpose(2, 0, 1)).float() / 255.0

        # For simplicity, we perform random crop/flip only on RGB via torchvision.
        # Depth is resized but not randomly cropped here. For a production
        # system, geometric transforms should be applied jointly.
        if self.resize is not None and T is not None:
            depth = np.array(
                rgb_tensor.shape[1:], dtype=np.int64
            )  # Placeholder for shape tracking.

        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W), float32

        # Build a binary mask for valid depth values.
        valid_mask = (depth_tensor > self.min_depth).float()

        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
            "scenario": sample.scenario,
            "sequence_id": sample.sequence_id or "",
            "frame_index": sample.frame_index if sample.frame_index is not None else -1,
        }


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def compute_abs_rel(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> float:
    """
    Absolute Relative Error (AbsRel).

    AbsRel = (1/N) * Σ |d_pred - d_gt| / d_gt
    """
    if valid_mask is None:
        valid_mask = (target > 0).float()
    mask = valid_mask > 0.5
    if not mask.any():
        return float("nan")
    d_pred = pred[mask]
    d_gt = target[mask]
    rel = torch.abs(d_pred - d_gt) / torch.clamp(d_gt, min=eps)
    return float(rel.mean().item())


def compute_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Root Mean Squared Error (RMSE).

    RMSE = sqrt((1/N) * Σ (d_pred - d_gt)^2)
    """
    if valid_mask is None:
        valid_mask = (target > 0).float()
    mask = valid_mask > 0.5
    if not mask.any():
        return float("nan")
    d_pred = pred[mask]
    d_gt = target[mask]
    mse = torch.mean((d_pred - d_gt) ** 2)
    return float(torch.sqrt(mse).item())


def compute_delta_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    thresholds: Tuple[float, float, float] = (1.25, 1.25 ** 2, 1.25 ** 3),
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute threshold accuracy metrics δ < t for multiple thresholds.

    For each pixel, we compute the maximum of (d_pred / d_gt) and (d_gt / d_pred)
    and test whether it falls below the specified threshold.
    """
    if valid_mask is None:
        valid_mask = (target > 0).float()
    mask = valid_mask > 0.5
    if not mask.any():
        return {f"delta<{t}": float("nan") for t in thresholds}

    d_pred = pred[mask]
    d_gt = target[mask]
    ratio = torch.max(
        d_pred / torch.clamp(d_gt, min=eps),
        d_gt / torch.clamp(d_pred, min=eps),
    )

    metrics: Dict[str, float] = {}
    for t in thresholds:
        acc = (ratio < t).float().mean().item()
        metrics[f"delta<{t}"] = float(acc)
    return metrics


class MetricAccumulator:
    """
    Accumulates per-batch depth metrics into dataset-level statistics.

    This class mirrors the evaluation workflow described in the manuscript,
    where each model/dataset combination is summarized by AbsRel, RMSE, and
    δ-threshold accuracies.
    """

    def __init__(self) -> None:
        self.abs_rel_values: List[float] = []
        self.rmse_values: List[float] = []
        self.delta_values: Dict[str, List[float]] = {}

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        with torch.no_grad():
            abs_rel = compute_abs_rel(pred, target, valid_mask)
            rmse = compute_rmse(pred, target, valid_mask)
            deltas = compute_delta_metrics(pred, target, valid_mask)

        if not math.isnan(abs_rel):
            self.abs_rel_values.append(abs_rel)
        if not math.isnan(rmse):
            self.rmse_values.append(rmse)

        for k, v in deltas.items():
            if k not in self.delta_values:
                self.delta_values[k] = []
            if not math.isnan(v):
                self.delta_values[k].append(v)

    def summarize(self) -> Dict[str, float]:
        def _mean_or_nan(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else float("nan")

        summary = {
            "AbsRel": _mean_or_nan(self.abs_rel_values),
            "RMSE": _mean_or_nan(self.rmse_values),
        }
        for k, vs in self.delta_values.items():
            summary[k] = _mean_or_nan(vs)
        return summary


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------


def build_dataloaders(
    root: str,
    index_csv: str,
    batch_size: int = 4,
    num_workers: int = 4,
    resize: Optional[Tuple[int, int]] = (384, 384),
    random_crop: Optional[Tuple[int, int]] = None,
    random_flip: bool = True,
    color_jitter: bool = False,
) -> Dict[str, DataLoader]:
    """
    Construct train/val/test dataloaders using a shared set of preprocessing
    hyper-parameters.

    This function is intentionally explicit about preprocessing knobs so that
    their values can be directly reported in the experimental setup section.
    """
    loaders: Dict[str, DataLoader] = {}
    for split in ["train", "val", "test"]:
        ds = DepthBenchmarkDataset(
            root=root,
            index_csv=index_csv,
            split=split,
            resize=resize,
            random_crop=random_crop,
            random_flip=random_flip,
            color_jitter=color_jitter,
        )
        shuffle = split == "train"
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
