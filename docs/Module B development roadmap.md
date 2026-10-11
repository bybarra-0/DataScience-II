# Module B Development Roadmap: Classical Machine Learning Baselines

Authoritative source: [readme.md](file:///d:/F2026/DS2/Main%20Branch/readme.md)  
Module Owner: Erick  
Primary implementation file: [baselines.py](file:///d:/F2026/DS2/Main%20Branch/baselines.py)  
Dependency flow: Shared Infrastructure -> Module A -> Module B  

---

## 1. Objectives and Target Models

Module B benchmarks four classical algorithms against deep learning models on the imbalanced HAM10000 dataset:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machines (SVM)
4. Random Forest

### Performance Goals
- Exceed naive majority-class accuracy (67 percent).
- Maximize Balanced Multiclass Accuracy (macro recall) to handle the 58 to 1 class imbalance.
- Minimize false negatives on malignant lesions (melanoma and basal cell carcinoma).

---

## 2. Shared Infrastructure Integration

All models consume pre-extracted features and shared tools:

| Component | Source | Purpose |
|---|---|---|
| Feature Ingestion | [`get_classical_data`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L168) | Loads (N, 64) float32 PCA arrays and string labels |
| Constants and Seeds | [config.py](file:///d:/F2026/DS2/Main%20Branch/config.py) | Canonical `CLASS_NAMES`, fixed `SEED`, and file paths |
| Unified Evaluation | [`compute_metrics`](file:///d:/F2026/DS2/Main%20Branch/metrics.py#L10) | Computes balanced accuracy, macro F1, and per-class recall |
| Results Tracking | [`log_baseline_result`](file:///d:/F2026/DS2/Main%20Branch/baselines.py#L18) | Appends run metrics to [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) under `module="B"` |

---

## 3. Fast Prototyping Protocol via [`dummy_batch`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L371)

Test pipelines, estimator initialization, and grid search syntax in under one second without waiting for disk arrays:

```python
from preprocess import dummy_batch

# Generates synthetic (200, 64) float32 features and string labels
X_dummy, y_dummy = dummy_batch(batch_size=200, modality="classical")
```

Use `X_dummy` and `y_dummy` to verify that scikit-learn pipelines run cleanly before executing full training runs on real data.

---

## 4. Hardware Allocation and Evaluation Protocol

### Hardware Allocation
- Module B executes on local CPU workstations.
- Fitting classical estimators on 64-dimensional feature projections takes seconds on standard multi-core processors.

### Split Evaluation Policy (Leakage Control)
- Model fitting and hyperparameter optimization occur strictly on `split_name="train"` using 5-fold cross-validation.
- Model selection and configuration comparisons occur strictly on `split_name="val"`.
- The held-out test split (`split_name="test"`) is frozen until Checkpoint G3 and evaluated once for final reporting.

---

## 5. Development Phases

### Phase 1: Pipeline Construction and Feature Scaling
- Distance and margin estimators (KNN, SVM, Logistic Regression) require zero-mean, unit-variance features. Always wrap them in a pipeline with `StandardScaler`:
  ```python
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler

  model = make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=SEED))
  ```
- Tree-based estimators (Random Forest) do not require feature scaling.
- Apply `class_weight="balanced"` to Logistic Regression, SVM, and Random Forest to prevent majority-class collapse.

### Phase 2: Untuned Baseline Execution
- Train default configurations of all 4 models on `X_train`.
- Predict validation labels on `X_val`.
- Record runtimes and validation metrics into [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) via [`evaluate_and_log`](file:///d:/F2026/DS2/Main%20Branch/baselines.py#L31).

### Phase 3: Hyperparameter Optimization
- Set up 5-fold cross-validation using `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`.
- Use `GridSearchCV` scoring `balanced_accuracy` on the training split:
  - Logistic Regression: Inverse regularization strength `C` and penalty types.
  - KNN: Number of neighbors `n_neighbors` (sweep 3 to 25) and distance weights (`uniform` vs `distance`).
  - SVM: Regularization `C` and kernel types (`linear`, `rbf`).
  - Random Forest: `n_estimators`, `max_depth`, and `min_samples_split`.
- Evaluate best estimators on `X_val` and log tuned rows to [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv).

### Phase 4: Error Analysis and Visual Deliverables
- Generate confusion matrix heatmaps using `seaborn` and `matplotlib` for the best model per architecture.
- Save plots to `figures/` (such as `figures/cm_random_forest.png`).

---

## 6. Deliverables Checklist

- [ ] [baselines.py](file:///d:/F2026/DS2/Main%20Branch/baselines.py) fully implemented for all four estimators.
- [ ] Populated rows in [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) comparing default vs tuned baselines.
- [ ] Exported confusion matrix heatmaps in `figures/`.
- [ ] Written baseline methodology and results discussion drafted for the final report.
