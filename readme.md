# Multi-Class Skin Lesion Classification Benchmark (HAM10000)

Comparative evaluation of classical machine learning algorithms against deep convolutional neural networks for automated skin lesion diagnosis across seven diagnostic categories.

---

## 1. Project Overview

This repository benchmarks classical estimators against deep learning architectures using the HAM10000 dataset. The workflow is divided into three functional modules: Module A (data preprocessing and feature engineering), Module B (classical baseline models), and Module C (deep learning models).

### Dataset Summary

| Attribute | Details |
|---|---|
| Dataset | HAM10000 (Human Against Machine with 10000 training images) |
| Source | Harvard Dataverse (https://doi.org/10.7910/DVN/DBW86T) |
| License | CC BY-NC 4.0 |
| Total Images | 10015 dermatoscopic images (600x450 RGB) |
| Unique Lesions | 7470 unique lesions |
| Diagnostic Classes | 7 classes: akiec, bcc, bkl, df, mel, nv, vasc |
| Class Imbalance | 58 to 1 ratio (nv: 6705 vs df: 115) |
| Partitioning Rule | Grouped strictly on `lesion_id` to eliminate cross-split patient leakage |
| Primary Metric | Balanced Multiclass Accuracy (macro recall / mean per-class sensitivity) |

---

## 2. Repository Layout

```
.
|-- config.py                 # Shared constants, paths, class order, and seeds
|-- data.py                   # Metadata parsing, split verification, and class weights
|-- preprocess.py             # Feature extraction, PCA caching, and PyTorch streaming
|-- baselines.py              # Classical ML estimators (Module B)
|-- model.py                  # Custom CNN architecture (Module C)
|-- pretrained.py             # Pretrained EfficientNetB0 transfer pipeline (Module C)
|-- metrics.py                # Unified evaluation metric computation
|-- results.csv               # Central benchmark experiment log
|-- requirements.txt          # Pinned project dependencies
|-- splits/
|   `-- splits.csv            # Frozen 70/15/15 lesion-grouped split manifest
|-- figures/                  # Generated plots and confusion matrices
`-- docs/
    |-- Module A development roadmap.md   # Architectural specifications for Module A
    |-- Module B development roadmap.md   # Baseline modeling specs and tuning guide
    |-- Module C development roadmap.md   # Deep learning and transfer learning guide
    |-- Project Proposal.md               # Project proposal and experimental scope
    `-- project_management.md             # Writing map, checkpoints, and deliverables
```

---

## 3. Environment and Setup

### Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Ingestion

1. Download `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` from Harvard Dataverse.
2. Place archives in `data/` or extract all 10015 images into `data/images/`.
3. Verify file integrity and partition alignment:

```bash
python data.py --verify
```

---

## 4. Module Execution and API Reference

### Module A: Preprocessing and Data Pipeline

Status: Fully Implemented

Module A provides leak-free data cleaning, visual deliverables, on-disk PCA feature caching for classical models, and a streaming PyTorch DataLoader pipeline for deep architectures.

#### CLI Commands

```bash
# Generate exploratory visual deliverables in figures/
python preprocess.py --figures

# Extract and cache 64-component PCA feature matrices in data/
python preprocess.py --features

# Execute geometric resize strategy benchmark
python preprocess.py --benchmark
```

#### Module B Consumer Interface (Classical Models)

Function [`get_classical_data`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L168) provides 64-component PCA features derived from 64x64 grayscale images. The PCA transformer is fitted strictly on the training partition and cached to `models/baselines/pca_64.joblib` to prevent data leakage. Extracted feature matrices are saved to `data/classical_features_{split_name}.npz`.

```python
from preprocess import get_classical_data

# Loads (N, 64) float32 PCA features and (N,) string targets
X_train, y_train = get_classical_data(split_name="train")
X_val, y_val = get_classical_data(split_name="val")
```

#### Module C Consumer Interface (Deep Models)

Function [`get_deep_loader`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L268) wraps [`HAM10000Dataset`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L245) to stream 224x224 RGB images on the fly, avoiding high RAM usage. Augmentations (rotations, flips, affine scaling) are applied strictly to the training split.

```python
from preprocess import get_deep_loader

# Returns PyTorch DataLoader yielding (images, labels)
# images: torch.float32 of shape (B, 3, 224, 224), normalized via ImageNet parameters
# labels: torch.long of shape (B,) with integer targets [0, 6]
train_loader = get_deep_loader(split_name="train", batch_size=32, shuffle=True)
val_loader = get_deep_loader(split_name="val", batch_size=32, shuffle=False)
```

Detailed architectural decisions, label formatting rationale, and augmentation controls are documented in [docs/Module A development roadmap.md](file:///d:/F2026/DS2/Main%20Branch/docs/Module%20A%20development%20roadmap.md).

---

### Module B: Classical Baselines

Status: implemented - writing report

Entry point: [baselines.py](file:///d:/F2026/DS2/Main%20Branch/baselines.py)

Models evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Random Forest

All estimators consume features via [`get_classical_data`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L168) and log runs to [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) using [`compute_metrics`](file:///d:/F2026/DS2/Main%20Branch/metrics.py#L10).

Detailed baseline modeling instructions and tuning protocols are documented in [docs/Module B development roadmap.md](file:///d:/F2026/DS2/Main%20Branch/docs/Module%20B%20development%20roadmap.md).

---

### Module C: Deep Learning

Status: In Progress

Entry points: [model.py](file:///d:/F2026/DS2/Main%20Branch/model.py) and [pretrained.py](file:///d:/F2026/DS2/Main%20Branch/pretrained.py)

Architectures evaluated:
- Custom Convolutional Neural Network
- Transfer learning with pretrained EfficientNetB0

Models consume streaming batches via [`get_deep_loader`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L268) and apply class-weighted loss via [`get_class_weights(as_tensor=True)`](file:///d:/F2026/DS2/Main%20Branch/data.py).

Detailed neural network architecture guidelines, training loops, and transfer learning instructions are documented in [docs/Module C development roadmap.md](file:///d:/F2026/DS2/Main%20Branch/docs/Module%20C%20development%20roadmap.md).

---

## 5. Benchmark Results

Experimental results logged in [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv):

| Model | Module | Config | Balanced Accuracy | Accuracy | Macro F1 | Runtime (s) |
|---|---|---|---|---|---|---|
| Random Forest | A | rf_resize_direct | 0.2740 | 0.6614 | 0.2752 | 1.88 |
| Random Forest | A | rf_resize_letterbox | 0.2519 | 0.6418 | 0.2498 | 1.95 |
| Random Forest | A | rf_resize_centercrop | 0.2704 | 0.6496 | 0.2777 | 1.98 |

---

## 6. Documentation Index

- [docs/Module A development roadmap.md](file:///d:/F2026/DS2/Main%20Branch/docs/Module%20A%20development%20roadmap.md): Complete engineering rationale, implementation phases, and leakage prevention rules for Module A
- [docs/Module B development roadmap.md](file:///d:/F2026/DS2/Main%20Branch/docs/Module%20B%20development%20roadmap.md): Baseline estimator pipelines, scaling rules, hyperparameter grids, and confusion matrix exports
- [docs/Module C development roadmap.md](file:///d:/F2026/DS2/Main%20Branch/docs/Module%20C%20development%20roadmap.md): Custom CNN architecture, EfficientNetB0 transfer learning, training loops, and loss curve exports
- [docs/Project Proposal.md](file:///d:/F2026/DS2/Main%20Branch/docs/Project%20Proposal.md): Project proposal, clinical motivation, and evaluation criteria
- [docs/project_management.md](file:///d:/F2026/DS2/Main%20Branch/docs/project_management.md): Final report writing map, checkpoint gates, course grading standards, and team responsibility breakdown
