## 1. Dataset decision record

```
Dataset chosen: HAM10000 (Human Against Machine with 10000 training images)
Link: https://doi.org/10.7910/DVN/DBW86T
License and redistribution terms: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)
Modality: Dermatoscopic RGB images (600x450 px) and tabular metadata (age, sex, anatomical site, diagnosis type)
Task and number of classes: Multi-class image classification (7 classes)
Approximate size (samples, disk): 10,015 images (representing 7,470 unique lesions), ~3.0 GB uncompressed (2.6 GB images + metadata)
Label definition:
- nv: Melanocytic nevi (benign)
- mel: Melanoma (malignant)
- bkl: Benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses)
- bcc: Basal cell carcinoma (malignant)
- akiec: Actinic keratoses and intraepithelial carcinoma / Bowen disease (pre-malignant / early-stage)
- vasc: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)
- df: Dermatofibroma (benign)
Grouping key (the ID that must not straddle splits, or "none, declared"): lesion_id
Official split available: no
Class distribution:
- nv: 6,705 (66.95%)
- mel: 1,113 (11.11%)
- bkl: 1,099 (10.97%)
- bcc: 514 (5.13%)
- akiec: 327 (3.27%)
- vasc: 142 (1.42%)
- df: 115 (1.15%)
Total: 10,015 images
Imbalance ratio (majority : minority): 58 to 1 (nv 6,705 : df 115)
Headline metric (set from class distribution): Balanced Multiclass Accuracy (mean per-class sensitivity / macro recall)
```

---

## 2. Shared infrastructure

| Item | File | Output |
|---|---|---|
| Project constants | `config.py` | Canonical class order, paths, image size, random seed |
| Data split and manifest loader | `data.py` + `splits/splits.csv` | One-time split generation with leakage check, manifest DataFrames |
| Shared metrics function | `metrics.py` | Unified metric set across all models |
| Results log | `results.csv` | Single summary table recording model performance |
| Pinned environment | `requirements.txt` | Reproducibility standard |

### `config.py`

Single source for constants shared across modules:

```python
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # canonical alphabetical order
SEED = 67
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (3, 224, 224)
DATA_ROOT, SPLIT_PATH, RESULTS_PATH
seed_everything(seed)   # seeds random, numpy, and pytorch
```

The alphabetical class order prevents axis misalignment in confusion matrices and per-class recall vectors across different models.

### `data.py`

```python
load_metadata() -> DataFrame              # one row per image, from HAM10000_metadata.csv
get_split(split=None) -> DataFrame        # "train" | "val" | "test", or full manifest
generate_split()                          # one-time generation script for splits/splits.csv
verify_images(auto_extract=True) -> bool  # validates all 10,015 files on disk
get_class_counts(split="train") -> dict   # class counts in canonical CLASS_NAMES order
get_class_weights(as_tensor=False)        # balanced inverse-frequency weights from train split
```

Returns index manifests with columns: `image_id, lesion_id, dx, dx_type, age, sex, localization, dataset, split`. File paths to images are resolved on the fly via `DATA_ROOT / "images" / f"{image_id}.jpg"`. Arrays are not materialized at this stage to avoid loading all images into memory simultaneously.

**Data ingestion and setup.** Raw images from Harvard Dataverse are distributed across two parts: `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`. Place both zip archives in `data/` or extract all contents into `data/images/` to yield a merged folder containing all 10,015 JPG files.

**Integrity verification.** `verify_images()` validates that all 10,015 images exist and are non-empty (>0 bytes). If `data/images/` is empty and zip files exist in `data/`, it automatically extracts and merges them into `data/images/`. Run `python data.py --verify` from the command line before training to verify setup.

**Class distribution and balanced weights.** `get_class_counts(split_name="train")` returns sample counts per class in canonical alphabetical order. `get_class_weights(as_tensor=False)` computes inverse-frequency weights strictly from the training split via w_c = N_train / (K * N_c). Setting `as_tensor=True` returns a `torch.float32` tensor directly consumable by Module C loss functions.

**Split policy.** HAM10000 publishes no official split. The 10,015 images cover 7,470 unique lesions, meaning roughly 2,545 images represent additional views of existing lesions. Grouping on `lesion_id` is mandatory to prevent identical lesions from appearing across training and testing sets. The split logic deduplicates by `lesion_id`, stratifies by `dx` at the lesion level (70% train, 15% validation, 15% test), maps back to all corresponding images, and asserts zero lesion overlap across splits.

**Static split artifact.** The split is generated once and committed as `splits/splits.csv`. All modules read directly from this static file to ensure identical splits across different environments.

**Validation and testing.** Model selection and hyperparameter tuning occur exclusively on the validation set. The test set is evaluated once at the conclusion of experiments.

### `metrics.py`

```python
compute_metrics(y_true, y_pred) -> dict
```

Calculates balanced accuracy, accuracy, macro precision, macro recall, macro F1, per-class recall, and confusion matrix.

**Headline metric: balanced multiclass accuracy** (mean per-class sensitivity / macro recall), selected to handle the 58 to 1 class imbalance. Macro F1 and per-class recall accompany it in every summary. Raw accuracy is tracked as secondary context.

### `results.csv`

Every model run logs one row containing `model_name, module, config, metrics, runtime_s`. This provides a direct source for the comparative tables and charts in the final report.

### `requirements.txt`

| Module | Package | Purpose |
|---|---|---|
| Shared Infrastructure | `numpy` | Array handling, seed management |
| Shared Infrastructure | `pandas` | Metadata loading, manifest indexing |
| Shared Infrastructure / B | `scikit-learn` | Stratified splitting, metrics, baseline models |
| Shared Infrastructure / B | `scipy` | Scientific computing support for scikit-learn |
| Module A (Preprocessing) | `Pillow`, `scikit-learn`, `torchvision` | Image loading, resizing, PCA feature extraction, dataset transforms |
| Module B / C (Reporting) | `matplotlib` | Loss curves and evaluation plots |
| Module B / C (Reporting) | `seaborn` | Confusion matrix heatmaps |
| Module C (Deep Learning) | `torch`, `torchvision` | CNN construction, EfficientNetB0 transfer learning |
| All Modules | `tqdm` | Progress tracking for image loops and batch processing |


---

## 3. Open modules

**Dependency:** Shared Infrastructure -> Module A -> Module B + C

### Module A: Preprocessing and data loading

- Dataset acquisition, cleaning, and metadata parsing
  - Unpack archives `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip` into `DATA_ROOT / "images"`.
  - Validate that all 10,015 images exist and are readable using `data.verify_images()`.
  - Impute 57 missing values in the age column using the median value computed strictly from the training split.
  - Image files are mapped dynamically from image_id to DATA_ROOT / "images" / f"{image_id}.jpg".
  - Metadata cleaning vs feature scope:
    - Demographic metadata (age, sex, localization) is cleaned and verified in manifests strictly for exploratory data analysis (EDA), demographic distribution plots, and final report documentation.
    - Model feature inputs for Modules B and C are strictly image-derived (64 PCA components for classical models, RGB tensors for deep models) to maintain an unconfounded, vision-only benchmark.
- Classical feature representation for Module B
  - `preprocess.get_classical_data(split_name: str = "train", force_recompute: bool = False)` returns `(X, y)` where `X` is an `(N, 64)` float32 array and `y` is an `(N,)` string label array matching `CLASS_NAMES`.
  - Images are downsampled to 64x64 grayscale. When `split_name == "train"`, a 64-component PCA model is fitted strictly on the training set and saved to `models/baselines/pca_64.joblib`. Validation and test sets are projected using this saved PCA transformer to prevent data leakage.
  - On-disk feature caching: Extracted arrays are automatically serialized to `data/classical_features_{split_name}.npz` via `np.savez_compressed(..., X=X, y=y)`. Subsequent calls load cached `.npz` arrays instantly. Pass `force_recompute=True` to invalidate the cache and re-extract from raw images.
- Deep model data pipeline for Module C
  - PyTorch `HAM10000Dataset` and `get_deep_loader(split_name, batch_size=32, shuffle=True)` stream image batches directly from disk on the fly.
  - Prevents loading all 10,015 images into RAM simultaneously, ensuring stable execution under memory constraints.
  - `get_deep_loader` returns a PyTorch `DataLoader` yielding batches of `(images, labels)`:
    - `images`: `torch.float32` tensor of shape `(batch_size, 3, 224, 224)` normalized with ImageNet mean and standard deviation.
    - `labels`: `torch.long` tensor of integer indices `0` to `6` mapped via `LABEL_TO_IDX`.
    - Integrates directly into `model.train_and_evaluate(model, model_name, config_desc, train_loader, val_loader)`.
- Label format contract and technical rationale
  - Classical baselines (`get_classical_data` -> string labels):
    - Scikit-learn native support: Classifiers in `baselines.py` (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`) handle string class targets directly.
    - Metrics alignment: `compute_metrics` in `metrics.py` evaluates performance using `labels=CLASS_NAMES`. Keeping labels as strings prevents confusion matrix axis misalignment.
    - Pipeline consistency: The synthetic generator `dummy_batch` for classical models returns string labels sampled from `CLASS_NAMES`. Maintaining string returns preserves interface consistency across baseline scripts.
  - Deep learning pipeline (`HAM10000Dataset` / `get_deep_loader` -> integer indices):
    - PyTorch loss requirement: `nn.CrossEntropyLoss` expects integer target tensors (`torch.long`) with values in range `[0, NUM_CLASSES - 1]`. Passing string labels produces a runtime error during training.
    - Batch collation: PyTorch default batch collation (`DataLoader`) bundles string outputs into a Python list of strings (`list[str]`), which cannot be transferred to GPU accelerator memory or evaluated by loss criteria.
- Augmentation strategy and leakage controls
  - Augmentation is restricted strictly to the training split (split == "train"). Validation and test sets use deterministic resizing and normalization only.
  - Applied transforms preserve dermatoscopic validity: random horizontal flip, random vertical flip, random 90-degree rotations, and slight affine scaling.
  - Excessive color jittering is omitted to preserve diagnostic color variations.
  - Clinical artifacts (hairs, gel bubbles, peripheral vignetting) documented in HAM10000 are preserved to evaluate realistic diagnostic robustness rather than introducing synthetic masking distortions.
- Class balance handling and loss weights
  - Single programmatic source: `data.get_class_counts(split_name="train")` and `data.get_class_weights(as_tensor=True)` compute sample frequencies and inverse-frequency balanced weights strictly on the training split to prevent leakage.
  - Module A deliverable: Uses `get_class_counts()` to generate `figures/class_distribution.png` documenting sample counts across classes and splits.
  - Module B consumption: Classical estimators apply `class_weight="balanced"` to automatically weight training samples.
  - Module C consumption: Passes `data.get_class_weights(as_tensor=True)` into PyTorch `nn.CrossEntropyLoss(weight=weights)`.
- Experimental benchmark for Module A
  - Compare 3 image resize strategies to evaluate geometric impact on classification:
    1. Direct resize: Stretch 600x450 RGB to 224x224 (distorts aspect ratio).
    2. Aspect-ratio preserving padding: Scale maintaining 4:3 aspect ratio, with black letterbox borders to 224x224.
    3. Center crop: Crop 450x450 center square from 600x450, downsample to 224x224.
  - Standardized feature extraction per strategy:
    - Downsample resized 224x224 images to 64x64 grayscale.
    - Fit a separate 64-component PCA model strictly on the training split for each strategy.
    - Project the validation split to produce an (N_val, 64) feature matrix.
  - Fixed baseline evaluation:
    - Train an identical RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED) on each feature set.
    - Evaluate on the validation split with compute_metrics.
    - Log 3 comparison rows to results.csv under module "A" with fit runtimes and validation metrics (configs: rf_resize_direct, rf_resize_letterbox, rf_resize_centercrop).
- Visual deliverables
  - Save `figures/sample_inspection.png` displaying a grid of representative images for all 7 lesion classes.
  - Save `figures/class_distribution.png` showing sample counts across classes and splits.

### Module B: Classical baselines

- 4 models specified in the proposal: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forests. Apply `class_weight="balanced"` to Logistic Regression, SVM, and Random Forest to prevent majority-class collapse.
- Feature standardization: Wrap estimators in `StandardScaler` pipelines to normalize inputs for distance-based and margin-based models.
- Hyperparameter search: Tune parameters using Stratified K-Fold cross-validation on the training split, optimizing for `balanced_accuracy`.
- Results and visualization: Append run metrics and fit runtimes to `results.csv`. Export confusion matrix heatmaps to `figures/`.

Input representation comes from Module A via `preprocess.get_classical_data()`. Writes no split logic and no metric of its own.

### Module C: Deep model

- Custom convolutional neural network (CNN) constructed in PyTorch.
- Transfer learning pipeline with a fine-tuned EfficientNetB0 backbone in PyTorch.
- Training loops, hyperparameters, learning rate scheduling, and overfitting controls.
- Training and validation loss and accuracy curves exported to `figures/`.
- Run metrics appended to `results.csv` and confusion matrix heatmaps saved to `figures/`.

---

## 4. Writing map

Per the proposal requirements, "You are expected to share the workload evenly, and every group member is expected to participate in both the experiments and writing. (As a group, you only need to submit one proposal and one report, though. So, you need to work together and coordinate your efforts.)"

Therefore, TA expects each of us to have parts in writing, parts in coding.

## Proposal writing map

| Proposal section     | Drafted by                                                   |
| :------------------- | ------------------------------------------------------------ |
| 1. Introduction      | Erick                                                        |
| 2. Motivation        | each write a sentence or paragraph of their own motivation. Edward merges |
| 3.  Evaluation       | Edward                                                       |
| 4. Resources         | each write the resources needed for their own module. Edward merges |
| 5. Team contribution | written together                                             |

## Final Report writing map	

| Final Report section | Drafted by |
|:--|---|
| Abstract | Edward, written last |
| 1. Introduction | ? |
| 2. Related Work | ? |
| 3. Proposed Method: dataset and preprocessing | Module A owner |
| 3. Proposed Method: baseline models | Module B owner |
| 3. Proposed Method: deep model | Module C owner |
| 4. Experiments | each writes their own setup |
| 5. Results and Discussion | each analyzes their own models from results assembled by Edward |
| 6. Conclusions | Edward |
| 7. Team Contributions | Written together, updated at the end |

Edward sets up the final report document skeleton, headings, figure placeholders, citation style, and does the final editorial pass format consistency. Everyone drafts into that skeleton.

---

## 5. Checkpoints

| Gate | Target | Must exist |
|---|---|---|
| G1 | Before proposal due on Sep 18 | Dataset downloaded and loading, one sample batch inspected |
| G2 | Before 10/15 | Split and metrics frozen, `splits/splits.csv` committed. B and C each have one model training on dummy or real data. |
| G3 | Before 10/31 | Every model produces the full metric set on the real split |
| G4 | Two weeks before final report | Results table populated, all figures drawn |

A gate is passed by showing running code.

---

## 6. Rules for consistency

- One repo, one branch per person, no code emailed around. `Project Overview.md` lives in the repo, since it is the document that defines everyone's interfaces.
- One frozen split, committed as `splits/splits.csv` and grouped on `lesion_id`, and one metrics function, both written before any model is trained.
- Every result reported with the same metric set, logged to results.csv.
- Model selection happens on validation. The test split is opened once, at G3.

---

## 7. Deliverables

From the syllabus. Project is 30 percent of the final grade.

| Deliverable |
|---|
| Proposal, 1 to 3 pages, 800 to 1200 words, PDF |
| Presentation, recorded video + slides, 8 to 10 min plus 2 min Q and A |
| Written report, 6 to 8 pages excluding references |
| Poster |

**Reproducibility standard**: "The report must be detailed enough that a classmate could replicate your results without guesswork."

**Grading focus:** "Grades are based on your learning process, execution, and honest reporting, not on whether your model achieves high performance. If an experiment fails or yields low accuracy, you will not be penalized, provided you analyze the issues clearly and suggest future improvements."

---

## 8. Team Contributions draft (to fill in during the meeting)



## Member: Edward

- **Module:** Shared Infrastructure

- **Coding:** `config.py`, `data.py`, `metrics.py`, `results.py`, `verify_split.py`, the `preprocess.py` interface stub, `requirements.txt`

- **Experimental:** Split balance verification, leakage check, seed stability, pipeline sanity run

- **Writing:** Abstract, Introduction, Related Work, Evaluation Protocol (split and metrics), Results and Discussion synthesis, Conclusions, final editorial pass

  

## Member: Edward

- **Module A:** Preprocessing
- **Coding:** 
  - `preprocess.py`: metadata cleaning and age imputation, 64-component PCA feature extraction for Module B, PyTorch HAM10000Dataset and DataLoader for Module C, train-only augmentations.
- **Experimental:** Compare 3 resize strategies (direct resize vs aspect-ratio padding vs center crop) on a fixed Random Forest baseline, generate figures/sample_inspection.png and figures/class_distribution.png.
- **Writing:** Dataset description, Preprocessing methods, resize benchmark analysis, sample inspection figures.



## Member: Erick

- **Module B:** Classical baselines
- **Coding:** 
  - `baselines.py`: Logistic Regression, KNN, SVM, Random Forests
- **Experimental:** Hyperparameter search per model, k sweep for KNN, kernel comparison for SVM, runtime and memory measurement, etc.
- **Writing:** Baseline models methods, baseline experimental setup, baseline results analysis.



## Member: Bryan

- **Module C:** Deep Learning model
- **Coding:** 
  - `model.py`: custom CNN.
  - `pretrained.py` for pretrained EfficientNetB0.
- **Experimental:** Learning rate and batch size sweep, regularization or dropout comparison, from-scratch versus pretrained comparison, training and validation curves
- **Writing:** Deep model architecture, training setup, deep model results analysis
