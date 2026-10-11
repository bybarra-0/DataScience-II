# Module A Development Roadmap: Preprocessing and Data Loading

Authoritative source: [`readme.md`](file:///d:/F2026/DS2/Main%20Branch/readme.md) 
Module Owner: Edward 
Primary implementation file: [`preprocess.py`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) 
Dependency flow: Shared Infrastructure -> Module A -> Module B and Module C 

---

## 1. Module Overview and Objectives

Module A establishes the data foundation for all downstream benchmarking. It ingests the raw HAM10000 dataset, performs leak-free data cleaning, generates exploratory visual deliverables, provides a 64-component PCA feature pipeline for classical models in Module B, builds a streaming PyTorch DataLoader pipeline for deep models in Module C, and conducts an image resize benchmark.

### Core Targets
- Zero data leakage across splits grouped on `lesion_id`
- Deterministic feature pipelines reading from frozen manifest [`splits/splits.csv`](file:///d:/F2026/DS2/Main%20Branch/splits/splits.csv)
- Efficient on-disk caching to avoid out-of-memory crashes across 10,015 images
- Strict interface contracts matching requirements of [`baselines.py`](file:///d:/F2026/DS2/Main%20Branch/baselines.py) and [`model.py`](file:///d:/F2026/DS2/Main%20Branch/model.py)

---

## 2. Shared Infrastructure Integration and Dependencies

Module A consumes constants and utility routines from the shared infrastructure files:

| Source File | Imported Symbols | Purpose in Module A |
|---|---|---|
| [`config.py`](file:///d:/F2026/DS2/Main%20Branch/config.py) | [`CLASS_NAMES`](file:///d:/F2026/DS2/Main%20Branch/config.py#L17), [`SEED`](file:///d:/F2026/DS2/Main%20Branch/config.py#L27), [`IMAGE_SIZE`](file:///d:/F2026/DS2/Main%20Branch/config.py#L23), [`INPUT_SHAPE`](file:///d:/F2026/DS2/Main%20Branch/config.py#L24), [`DATA_ROOT`](file:///d:/F2026/DS2/Main%20Branch/config.py#L10), [`SPLIT_PATH`](file:///d:/F2026/DS2/Main%20Branch/config.py#L13), [`RESULTS_PATH`](file:///d:/F2026/DS2/Main%20Branch/config.py#L14), [`LABEL_TO_IDX`](file:///d:/F2026/DS2/Main%20Branch/config.py#L19), [`seed_everything`](file:///d:/F2026/DS2/Main%20Branch/config.py#L29) | Shared paths, canonical label order, seed control |
| [`data.py`](file:///d:/F2026/DS2/Main%20Branch/data.py) | [`load_metadata`](file:///d:/F2026/DS2/Main%20Branch/data.py#L5), [`get_split`](file:///d:/F2026/DS2/Main%20Branch/data.py#L54), [`verify_images`](file:///d:/F2026/DS2/Main%20Branch/data.py), [`get_class_counts`](file:///d:/F2026/DS2/Main%20Branch/data.py), [`get_class_weights`](file:///d:/F2026/DS2/Main%20Branch/data.py) | Manifest retrieval, file verification, class counts and loss weights |
| [`metrics.py`](file:///d:/F2026/DS2/Main%20Branch/metrics.py) | [`compute_metrics`](file:///d:/F2026/DS2/Main%20Branch/metrics.py#L10) | Evaluation of Module A resize benchmark models |
| [`results.csv`](file:///d:/F2026/DS2/Main%20Branch/results.csv) | Results logging table | Record benchmark metrics for resize strategies under `module="A"` |
| `joblib` | `joblib.dump`, `joblib.load` | Model persistence, serializing train-fitted pca transformer to disk and loading for leakage-free inference |

---

## 3. Development Phases

```
Phase 1: Ingestion and Cleaning
  Unpack images -> verify_images() -> Impute age via train median -> Map paths
       |
Phase 2: Visual Deliverables
  Export class_distribution.png -> Export sample_inspection.png
       |
Phase 3: Classical Pipeline (Module B)
  Downsample 64x64 gray -> Fit PCA(64) on train -> joblib.dump -> Cache .npz -> get_classical_data()
       |
Phase 4: Deep Pipeline (Module C)
  HAM10000Dataset -> Train augmentations -> ImageNet norm -> get_deep_loader()
       |
Phase 5: Resize Strategy Benchmark
  Direct vs Letterbox vs Center Crop -> Fit RF models -> Log to results.csv
       |
Phase 6: Final Report Writing
  Draft Section 3 dataset and preprocessing methods
```

### Phase 1: Data Ingestion, Integrity Verification, and Cleaning
1. Image archive extraction and verification:
   - Unpack `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` into `data/images/`.
   - Execute [`verify_images(auto_extract=True)`](file:///d:/F2026/DS2/Main%20Branch/data.py) to confirm all 10,015 images exist and are non-empty (>0 bytes).
2. Dynamic image path mapping:
   - Resolve image file paths dynamically via `DATA_ROOT / "images" / f"{image_id}.jpg"`.
   - Do not load full image arrays into memory simultaneously.
3. Metadata cleaning and missing value imputation:
   - Clean tabular metadata loaded via [`load_metadata()`](file:///d:/F2026/DS2/Main%20Branch/data.py#L5).
   - Impute 57 missing values in the `age` column using the median computed strictly from the training split (`split == "train"`) to prevent data leakage.
4. Feature scope separation:
   - Metadata columns (`age`, `sex`, `localization`) remain strictly for exploratory data analysis, distribution plots, and final report documentation.
   - Downstream models in Module B and Module C strictly receive image-derived inputs (PCA components or RGB pixel tensors) to guarantee an unconfounded, vision-only benchmark.

### Phase 2: Visual Deliverables
1. Class distribution plot:
   - Read split manifest via [`get_split()`](file:///d:/F2026/DS2/Main%20Branch/data.py#L54) and counts via [`get_class_counts()`](file:///d:/F2026/DS2/Main%20Branch/data.py).
   - Generate and save `figures/class_distribution.png` illustrating sample counts across all 7 classes broken down by split (train, val, test).
2. Dataset sample inspection grid:
   - Sample representative images for each of the 7 diagnostic classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
   - Generate and save `figures/sample_inspection.png` displaying an annotated grid of lesion examples.

### Phase 3: Classical Feature Pipeline for Module B
1. Technical rationale:
   - Flattening raw 224x224 RGB images produces 150,528 dimensions per image, which is computationally intractable for SVM and KNN and memory-prohibitive during cross-validation.
   - PCA reduces dimensionality to 64 components while preserving dominant variance.
2. Implementation steps for [`preprocess.get_classical_data`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L26):
   - Downsample images to 64x64 grayscale.
   - When `split_name == "train"`, fit a 64-component PCA model strictly on the training split.
   - Serialize the fitted PCA transformer to `models/baselines/pca_64.joblib` via `joblib.dump`.
   - When `split_name` is `"val"` or `"test"`, load the serialized transformer via `joblib.load` and project images without refitting to prevent leakage.
3. On-disk feature caching:
   - Save extracted `(X, y)` arrays to `data/classical_features_{split_name}.npz` via `np.savez_compressed(..., X=X, y=y)`.
   - On subsequent calls, check for the `.npz` archive and load instantly.
   - Support `force_recompute=True` parameter to invalidate cache and re-extract from raw files.
4. Label contract:
   - Return `X` as `(N, 64)` float32 array and `y` as `(N,)` 1D string array matching [`CLASS_NAMES`](file:///d:/F2026/DS2/Main%20Branch/config.py#L17).
   - Rationale: Scikit-learn estimators natively accept string targets, metrics evaluation with `labels=CLASS_NAMES` prevents axis misalignment, and interface consistency with [`dummy_batch`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L9) is maintained.

### Phase 4: Deep Learning Pipeline for Module C
1. Dataset implementation:
   - Construct PyTorch [`HAM10000Dataset`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) reading file paths from [`get_split(split_name)`](file:///d:/F2026/DS2/Main%20Branch/data.py#L54).
   - Stream images on demand from disk during training and inference to avoid RAM exhaustion.
2. Data augmentation strategy and leakage controls:
   - Apply augmentations strictly when `split == "train"`.
   - Preserved dermatoscopic transforms: random horizontal flip, random vertical flip, random 90-degree rotations, and slight affine scaling.
   - Omit aggressive color jitter to preserve diagnostic hue and saturation cues.
   - Preserve clinical artifacts (hairs, gel bubbles, peripheral vignetting) intact rather than applying algorithmic inpainting (such as DullRazor), testing realistic diagnostic robustness.
   - Validation and test splits use deterministic resizing and normalization only.
3. Normalization and tensor shape:
   - Resize images to 224x224 pixels.
   - Normalize RGB channels using ImageNet statistics: mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
   - Return float32 tensor of shape `(batch_size, 3, 224, 224)`.
4. Label contract:
   - Map string diagnostic labels to integer class indices `0` through `6` using [`LABEL_TO_IDX`](file:///d:/F2026/DS2/Main%20Branch/config.py#L19).
   - Return targets as `torch.long` tensors.
   - Rationale: PyTorch `nn.CrossEntropyLoss` requires integer target tensors; standard `DataLoader` batch collation converts strings into Python lists of strings, preventing GPU memory transfer and loss calculation.
5. DataLoader construction:
   - Implement [`get_deep_loader(split_name, batch_size=32, shuffle=True)`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py).
   - Return ready-to-consume PyTorch `DataLoader` directly compatible with `model.train_and_evaluate()`.

### Phase 5: Experimental Benchmark for Module A (Image Resize Strategies)
1. Geometric resize strategies:
   - Strategy 1: Direct resize - Stretch 600x450 RGB to 224x224 (distorts aspect ratio).
   - Strategy 2: Aspect-ratio preserving padding - Scale maintaining 4:3 ratio with black letterbox borders to 224x224.
   - Strategy 3: Center crop - Crop 450x450 center square from 600x450, downsample to 224x224.
2. Standardized feature extraction:
   - Downsample each resized 224x224 variant to 64x64 grayscale.
   - Fit a dedicated 64-component PCA model strictly on the training split for each strategy.
   - Project the validation split to produce an `(N_val, 64)` matrix per strategy.
3. Model evaluation:
   - Train an identical `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)` on each feature set.
   - Evaluate predictions on the validation split with [`compute_metrics`](file:///d:/F2026/DS2/Main%20Branch/metrics.py#L10).
4. Results logging:
   - Append 3 benchmark rows to [`results.csv`](file:///d:/F2026/DS2/Main%20Branch/results.csv) under `module="A"`:
     - Config 1: `rf_resize_direct`
     - Config 2: `rf_resize_letterbox`
     - Config 3: `rf_resize_centercrop`
   - Record balanced accuracy, accuracy, macro F1, macro precision, macro recall, and runtime in seconds.

### Phase 6: Final Report Writing
- Draft Final Report Section 3: "Proposed Method: dataset and preprocessing".
  - Dataset source, distribution, and 58 to 1 class imbalance.
  - Leak-free partitioning policy grouped on `lesion_id`.
  - Missing value imputation on `age`.
  - Grayscale 64-component PCA pipeline for classical estimators.
  - PyTorch on-the-fly streaming pipeline, train-only augmentations, and clinical artifact rationale.
  - Experimental analysis and comparison table of the 3 resize strategies.
  - Embed `figures/sample_inspection.png` and `figures/class_distribution.png`.

---

## 4. Technical Interface Contracts

| Function or Class | Arguments | Return Type | Consumer | Notes |
|---|---|---|---|---|
| [`transform`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L4) | `image_path: str` | `np.ndarray` of shape `(3, 224, 224)` | Shared image utility | Normalized float32 array |
| [`get_classical_data`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L26) | `split_name: str = "train"`, `force_recompute: bool = False` | `tuple[np.ndarray, np.ndarray]` -> `X: (N, 64)`, `y: (N,)` | Module B ([`baselines.py`](file:///d:/F2026/DS2/Main%20Branch/baselines.py)) | `X` is float32 PCA features, `y` is 1D string array matching [`CLASS_NAMES`](file:///d:/F2026/DS2/Main%20Branch/config.py#L17) |
| [`HAM10000Dataset`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) | `split_name: str`, `transform=None` | PyTorch `Dataset` | Module C ([`model.py`](file:///d:/F2026/DS2/Main%20Branch/model.py), [`pretrained.py`](file:///d:/F2026/DS2/Main%20Branch/pretrained.py)) | Streams images from disk, returns `(image_tensor, label_idx)` |
| [`get_deep_loader`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) | `split_name: str`, `batch_size: int = 32`, `shuffle: bool = True` | PyTorch `DataLoader` | Module C ([`model.py`](file:///d:/F2026/DS2/Main%20Branch/model.py), [`pretrained.py`](file:///d:/F2026/DS2/Main%20Branch/pretrained.py)) | Yields `(batch_images, batch_labels)`: images `(B, 3, 224, 224)` float32, labels `(B,)` int64 |

---

## 5. Artifact and File Outputs

| Deliverable | Output Path | Description |
|---|---|---|
| Cleaned Code Interface | [`preprocess.py`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) | Full implementation replacing dummy stubs |
| Serialized PCA Model | `models/baselines/pca_64.joblib` | 64-component PCA fitted strictly on training split, serialized via `joblib` |
| Cached Classical Features | `data/classical_features_train.npz`<br>`data/classical_features_val.npz`<br>`data/classical_features_test.npz` | Compressed NumPy feature caches |
| Class Distribution Figure | `figures/class_distribution.png` | Plot of sample counts across classes and splits |
| Sample Inspection Figure | `figures/sample_inspection.png` | 7-class lesion image grid |
| Benchmark Results | [`results.csv`](file:///d:/F2026/DS2/Main%20Branch/results.csv) | Logged rows for `rf_resize_direct`, `rf_resize_letterbox`, and `rf_resize_centercrop` |

---

## 6. Project Checkpoint Gates

| Gate | Target Date | Gate Requirement | Module A Role |
|---|---|---|---|
| G1 | Before 09/18 | Dataset downloaded and loading, sample batch inspected | Image acquisition, extraction verification, initial batch loading |
| G2 | Before 10/15 | Split and metrics frozen, baseline and deep models training on dummy or real data | Deliver [`get_classical_data`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L26) and [`get_deep_loader`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) to unblock real training in Module B and Module C |
| G3 | Before 10/31 | Every model produces full metric set on real split | Run resize strategy benchmark on validation split, log metrics to [`results.csv`](file:///d:/F2026/DS2/Main%20Branch/results.csv), open test split once |
| G4 | Two weeks before final report | Results table populated, all figures drawn | Verify all Module A rows in [`results.csv`](file:///d:/F2026/DS2/Main%20Branch/results.csv), complete `figures/sample_inspection.png` and `figures/class_distribution.png`, draft Section 3 text |
