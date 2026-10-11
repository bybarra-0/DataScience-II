# Module C Development Roadmap: Deep Learning Architectures

Authoritative source: [readme.md](file:///d:/F2026/DS2/Main%20Branch/readme.md)  
Module Owner: Bryan  
Primary implementation files: [model.py](file:///d:/F2026/DS2/Main%20Branch/model.py) and [pretrained.py](file:///d:/F2026/DS2/Main%20Branch/pretrained.py)  
Dependency flow: Shared Infrastructure -> Module A -> Module C  

---

## 1. Objectives and Target Models

Module C benchmarks convolutional neural networks against classical baselines on the imbalanced HAM10000 dataset:

1. Custom CNN: Built from scratch in PyTorch to establish a native deep baseline.
2. Transfer Learning: Fine-tuned EfficientNetB0 backbone pre-trained on ImageNet.

### Performance Goals
- Exceed classical baseline performance across all seven diagnostic categories.
- Achieve Balanced Multiclass Accuracy above 70 percent on held-out splits.
- Attain sensitivity (per-class recall) above 75 percent on malignant classes (melanoma and basal cell carcinoma).

---

## 2. Shared Infrastructure Integration

All models stream batches on the fly and log results to central tracking:

| Component | Source | Purpose |
|---|---|---|
| Data Streaming | [`get_deep_loader`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L268) | Yields batches of (B, 3, 224, 224) float32 tensors and (B,) int64 targets |
| Loss Weighting | [`get_class_weights`](file:///d:/F2026/DS2/Main%20Branch/data.py) | Generates `torch.float32` inverse-frequency weights for `nn.CrossEntropyLoss` |
| Constants and Seeds | [config.py](file:///d:/F2026/DS2/Main%20Branch/config.py) | Canonical `CLASS_NAMES`, `NUM_CLASSES`, `INPUT_SHAPE`, and [`seed_everything`](file:///d:/F2026/DS2/Main%20Branch/config.py#L29) |
| Unified Evaluation | [`compute_metrics`](file:///d:/F2026/DS2/Main%20Branch/metrics.py#L10) | Evaluates integer predictions converted via `torch.argmax(logits, dim=1)` |
| Results Tracking | [`log_deep_result`](file:///d:/F2026/DS2/Main%20Branch/model.py#L19) | Appends run metrics to [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) under `module="C"` |

---

## 3. Fast Prototyping Protocol via [`dummy_batch`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L371)

Verify neural network layer connectivity, tensor shapes, and loss backward passes in under one second without waiting for full dataset downloads:

```python
from preprocess import dummy_batch

# Generates synthetic (32, 3, 224, 224) float32 tensors and string labels
X_dummy, y_dummy = dummy_batch(batch_size=32, modality="deep")
```

Use `X_dummy` to test that forward passes through [`CustomCNN`](file:///d:/F2026/DS2/Main%20Branch/model.py#L35) output logits of shape `(32, 7)` before executing training runs.

---

## 4. Hardware Allocation and Evaluation Protocol

### Hardware Allocation
- Module C requires GPU acceleration.
- Use cloud-based GPU runtimes (such as Google Colab T4 or V100 instances).
- Training deep models locally on CPU workstations across 10015 RGB images will create severe runtime bottlenecks.

### Split Evaluation Policy (Leakage Control)
- Network training, learning rate schedules, and checkpoint selection occur strictly on `split_name="train"` and `split_name="val"`.
- The held-out test split (`split_name="test"`) is frozen until Checkpoint G3 and evaluated once for final reporting.

---

## 5. Development Phases

### Phase 1: Custom CNN Architecture Design
- Define [`CustomCNN`](file:///d:/F2026/DS2/Main%20Branch/model.py#L35) in [model.py](file:///d:/F2026/DS2/Main%20Branch/model.py).
- Build stacked convolutional blocks: `Conv2d`, `BatchNorm2d`, `ReLU`, and `MaxPool2d`.
- Apply `Dropout` (rates between 0.2 and 0.5) before the final fully connected layer to prevent overfitting on the imbalanced dataset.
- Output raw unnormalized logits of dimension `(batch_size, NUM_CLASSES)`.

### Phase 2: EfficientNetB0 Transfer Learning Pipeline
- In [pretrained.py](file:///d:/F2026/DS2/Main%20Branch/pretrained.py), instantiate `torchvision.models.efficientnet_b0(weights=...)`.
- Replace the final classifier layer:
  ```python
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
  ```
- Compare two transfer strategies:
  1. Feature extraction: Freeze convolutional backbone parameters (`requires_grad = False`) and train only the head.
  2. Fine-tuning: Train the entire network with a small learning rate (such as `1e-4`).

### Phase 3: Training Loop and Class Imbalance Handling
- Pass inverse-frequency training weights into the loss criterion:
  ```python
  weights_tensor = get_class_weights(as_tensor=True).to(device)
  criterion = nn.CrossEntropyLoss(weight=weights_tensor)
  ```
- Implement `train_one_epoch` and `evaluate` in [model.py](file:///d:/F2026/DS2/Main%20Branch/model.py).
- Track validation balanced accuracy per epoch and save best checkpoints (`.pt`) to `models/deep/`.

### Phase 4: Training Curves and Error Analysis
- Record training loss, validation loss, and validation balanced accuracy across epochs.
- Implement `plot_training_curves` to export loss and accuracy curves to `figures/`.
- Export confusion matrix heatmaps for the best custom CNN and fine-tuned EfficientNetB0 models.
- Append final run metrics to [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) via [`log_deep_result`](file:///d:/F2026/DS2/Main%20Branch/model.py#L19).

---

## 6. Deliverables Checklist

- [ ] [model.py](file:///d:/F2026/DS2/Main%20Branch/model.py) (Custom CNN) fully implemented with training loops.
- [ ] [pretrained.py](file:///d:/F2026/DS2/Main%20Branch/pretrained.py) (EfficientNetB0) fully implemented for transfer learning.
- [ ] Saved model checkpoint files (`.pt`) stored in `models/deep/`.
- [ ] Populated benchmark rows in [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) under `module="C"`.
- [ ] Training loss/accuracy curves and confusion matrix heatmaps saved to `figures/`.
- [ ] Deep learning methodology, architecture specifications, and results analysis drafted for the final report.
