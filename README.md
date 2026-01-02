# Performance Benchmarking and Optimization Strategies for Depth Estimation Algorithms in Unstructured Environments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

This repository contains the reference implementation and experimental pipeline for the research paper:

**Performance Benchmarking and Optimization Strategies for Depth Estimation Algorithms in Unstructured Environments**  
_Yuhan Li_  
_Northeastern University, 2025_

## 📑 Table of Contents

-  [Project Goals](#-project-goals)
-  [Overview](#-overview)
-  [Repository Structure](#-repository-structure)
-  [Supported Algorithms](#-supported-algorithms)
-  [Evaluation Metrics](#-evaluation-metrics)
-  [Datasets & Scenarios](#-datasets--scenarios)
-  [Running Experiments](#-running-experiments)
-  [Reproducibility](#-reproducibility)
-  [Key Findings](#-key-findings-summary)
-  [Citation](#-citation)
-  [License](#-license)

## 🎯 Project Goals

This project provides a **deployment-oriented, reproducible benchmarking framework** for depth estimation algorithms, jointly evaluating:

-  ✅ Estimation accuracy
-  ⚡ Inference latency and throughput
-  💾 GPU memory usage and utilization
-  🛡️ Robustness to environmental degradations

Designed for real-world robotics and autonomous systems deployment across diverse GPU platforms.

## 📖 Overview

Depth estimation is a critical perception capability for autonomous robotic systems operating in complex, unstructured environments. While prior work primarily emphasizes accuracy on static benchmarks, **real-world deployment requires simultaneous consideration of multiple factors**.

### Key Challenges Addressed

| Challenge                | Solution                                      |
| ------------------------ | --------------------------------------------- |
| Accuracy-only benchmarks | Multi-metric evaluation framework             |
| Platform-agnostic models | Cross-platform performance profiling          |
| Opaque trade-offs        | Systematic accuracy–latency–resource analysis |
| Deployment uncertainty   | Data-driven algorithm–hardware matching       |

### Framework Capabilities

-  📊 **Fair comparison** of representative depth estimation architectures
-  🖥️ **Cross-platform profiling** (embedded → datacenter GPUs)
-  ⚖️ **Trade-off identification** for accuracy, latency, and resources
-  🎯 **Data-driven recommendations** for real-world deployment

## 📁 Repository Structure

```
.
├── code/
│   ├── depth_benchmark_dataset_and_metrics.py
│   │   └── Dataset abstractions, scenario taxonomy, preprocessing, and
│   │       depth accuracy metrics (AbsRel, RMSE, δ thresholds)
│   │
│   ├── depth_model_zoo_and_factories.py
│   │   └── Model zoo with representative architectures: CNN baseline,
│   │       lightweight CNN, transformer, and hybrid CNN–Transformer
│   │
│   ├── depth_training_and_evaluation_pipeline.py
│   │   └── Unified training/evaluation pipeline with multi-metric logging
│   │       and scenario-aware dataset handling
│   │
│   ├── depth_inference_performance_profiler.py
│   │   └── Inference profiling (latency, throughput, memory, GPU
│   │       utilization, power consumption)
│   │
│   ├── depth_hyperparameter_and_resolution_ablation.py
│   │   └── Automated ablation studies for learning rates, weight decay,
│   │       and input resolution
│   │
│   ├── depth_robustness_under_environmental_conditions.py
│   │   └── Robustness evaluation under simulated environmental
│   │       degradations (illumination, noise, blur, occlusion)
│   │
│   ├── depth_uncertainty_and_calibration_analysis.py
│   │   └── MC-dropout uncertainty estimation and calibration analysis
│   │       with reliability diagnostics
│   │
│   └── depth_results_aggregation_and_latex_export.py
│       └── CSV aggregation and LaTeX table generation for publication
│
├── data/
│   └── Benchmark datasets and synthetic test data
│
└── docs/
    └── Generated plots, tables, and diagrams (referenced in paper)
```

## 🤖 Supported Algorithms

The benchmarking framework includes representative depth estimation families commonly used in robotics and autonomous systems:

| Architecture               | Description                       | Use Case                   |
| -------------------------- | --------------------------------- | -------------------------- |
| **CNN Baseline**           | UNet-style encoder–decoder        | Baseline comparison        |
| **Lightweight CNN**        | Depthwise separable convolutions  | Edge/embedded deployment   |
| **Transformer**            | Global context via self-attention | High-accuracy applications |
| **Hybrid CNN–Transformer** | Combined local + global features  | Balanced performance       |
| **Stereo variants**        | Supervised and self-supervised    | Multi-view scenarios       |

### Uniform Interface

All models expose a consistent forward interface for easy benchmarking:

```python
forward(rgb: Tensor[B, 3, H, W]) -> Tensor[B, 1, H, W]
```

## 📊 Evaluation Metrics

The framework implements the full metric taxonomy described in the paper:

### Accuracy Metrics

-  **Absolute Relative Error (AbsRel)**
-  **Root Mean Squared Error (RMSE)**
-  **Threshold accuracies**: δ < 1.25, 1.25², 1.25³

### Performance Metrics

-  Single-frame inference latency (ms)
-  Throughput (FPS)
-  GPU memory footprint (MB)

GPU utilization and power draw (when supported)

Robustness & Reliability

Performance under synthetic environmental corruptions

Uncertainty–error correlation via MC dropout

Reliability-style calibration analysis

## 🗂️ Datasets & Scenarios

The framework supports scenario-aware evaluation across diverse unstructured environments:

| Scenario                 | Description                                 |
| ------------------------ | ------------------------------------------- |
| 🏠 **Indoor structured** | Organized indoor spaces with clear geometry |
| 🏚️ **Indoor cluttered**  | Complex indoor scenes with occlusions       |
| 🌳 **Outdoor simple**    | Open outdoor environments                   |
| 🏙️ **Outdoor complex**   | Dense urban or natural outdoor scenes       |

**Ground-truth depth** is obtained from LiDAR or RGB-D sensors with proper calibration.

**Public datasets** commonly used: NYU Depth V2, SUN RGB-D, ETH3D, and related RGB-D benchmarks (see paper for full list).

## 🚀 Running Experiments

### 1️⃣ Training a Model

```bash
python depth_training_and_evaluation_pipeline.py \
  --arch transformer \
  --dataset-root /path/to/dataset \
  --index-csv index.csv \
  --batch-size 4 \
  --max-epochs 40
```

### 2️⃣ Profiling Inference Performance

```bash
python depth_inference_performance_profiler.py \
  --arch transformer \
  --height 384 \
  --width 384
```

### 3️⃣ Robustness Evaluation

```bash
python depth_robustness_under_environmental_conditions.py \
  --arch transformer \
  --checkpoint best_transformer.pth \
  --dataset-root /path/to/dataset \
  --index-csv index.csv
```

### 4️⃣ Aggregating Results & Exporting LaTeX Tables

```bash
python depth_results_aggregation_and_latex_export.py
```

## ♻️ Reproducibility

✅ All experiments are designed to run in containerized environments  
✅ Explicit configuration logging ensures exact reproducibility  
✅ Metric definitions and preprocessing pipelines are fully transparent  
✅ Results are aggregated using deterministic evaluation protocols

### Recommended Stack

-  **OS**: Ubuntu 22.04
-  **CUDA**: 12.x
-  **PyTorch**: ≥ 2.0
-  **TensorRT**: Optional (for optimized inference)

## 📈 Key Findings (Summary)

-  🎯 Transformer-based models achieve **12–18% lower AbsRel** than CNN baselines but incur **2–3× higher latency**
-  ⚡ Lightweight CNNs enable **real-time performance** on embedded GPUs with modest accuracy trade-offs
-  📊 Algorithm rankings **vary significantly by hardware platform**
-  🚀 Multi-task learning and TensorRT optimizations provide **substantial efficiency gains**
-  ⚖️ Accurate deployment decisions require **joint consideration** of accuracy, latency, and resource usage

## 📝 Citation

If you use this code or benchmarking framework in your research, please cite:

```bibtex
@article{li_depth_benchmarking_2025,
  title={Performance Benchmarking and Optimization Strategies for Depth Estimation Algorithms in Unstructured Environments},
  author={Li, Yuhan},
  institution={Northeastern University},
  year={2025}
}
```

## 📄 License

This project is released for **research and academic use**.  
Please see individual dataset licenses for data usage terms.

---

<div align="center">

**Made with ❤️ by Yuhan Li @ Northeastern University**

[Report Bug](https://github.com/CatNinjaLuna/Depth_Estimation/issues) · [Request Feature](https://github.com/CatNinjaLuna/Depth_Estimation/issues)

</div>
