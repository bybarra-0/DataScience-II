## 1. Dataset decision record

```
Dataset chosen: HAM10000 (Human Against Machine with 10000 training images)
Link: https://doi.org/10.7910/DVN/DBW86T
License and redistribution terms: CC BY-NC 4.0
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
DATA_ROOT, SPLIT_PATH, RESULTS_PATH
seed_everything(seed)   # seeds random, numpy, and deep learning framework
```

The alphabetical class order prevents axis misalignment in confusion matrices and per-class recall vectors across different models.

### `data.py`

```python
load_metadata() -> DataFrame          # one row per image, from HAM10000_metadata.csv
get_split(split=None) -> DataFrame    # "train" | "val" | "test", or full manifest
generate_split()                      # one-time generation script for splits/splits.csv
```

Returns index manifests with columns: `image_id, lesion_id, dx, dx_type, age, sex, localization, filepath, split`. Arrays are not materialized at this stage to avoid loading all images into memory simultaneously.

**Split policy.** HAM10000 publishes no official split. The 10,015 images cover 7,470 unique lesions, meaning roughly 2,545 images represent additional views of existing lesions. Grouping on `lesion_id` is mandatory to prevent identical lesions from appearing across training and testing sets. The split logic deduplicates by `lesion_id`, stratifies by `dx` at the lesion level (70% train, 15% validation, 15% test), maps back to all corresponding images, and asserts zero lesion overlap across splits.

**Static split artifact.** The split is generated once and committed as `splits/splits.csv`. All modules read directly from this static file to ensure identical splits across different environments.

**Validation and testing.** Model selection and hyperparameter tuning occur exclusively on the validation set. The test set is evaluated once at the conclusion of experiments.

### `metrics.py`

```python
compute_metrics(y_true, y_pred, y_proba=None) -> dict
```

Calculates balanced accuracy, macro precision, macro recall, macro F1, per-class recall, and confusion matrix. Multi-class AUROC is computed when predicted probabilities are supplied.

**Headline metric: balanced multiclass accuracy** (mean per-class sensitivity / macro recall), selected to handle the 58 to 1 class imbalance. Macro F1 and per-class recall accompany it in every summary. Raw accuracy is tracked as secondary context.

### `results.csv`

Every model run logs one row containing `model_name, module, config, metrics, runtime_s`. This provides a direct source for the comparative tables and charts in the final report.

---

## 3. Open modules

**Dependency:** Shared Infrastructure -> Module A -> Module B + C

### Module A: Preprocessing and data loading

- Dataset acquisition, cleaning, and any parsing of annotation files
- Conversion to a fixed-shape model input (images: resize and normalize. audio: segment, length-normalize, convert to spectrogram or extract features. tabular: encode and scale)
- Augmentation appropriate to the modality
- Class balance handling and documentation of what was done
- Sample inspection, at least one figure showing what the data looks like
- Agrees with Module B on the feature representation classical models will consume, since it is usually not the same tensor Module C takes

Module A implements `preprocess.transform()`. Modules B and C build against the interface stub using dummy data initially, then swap in real data when A finishes. The stub itself, meaning the frozen signatures plus a `dummy_batch()` that returns correctly shaped random tensors and labels, ships with the shared infrastructure block and lands before Module A is implemented, so B and C are never blocked on preprocessing to reach G2.

### Module B: Classical baselines

- 3 to 4 models among: Logistic Regression, Random Forest, SVM, KNN, Decision Tree.
- Hyperparameter search for each
- Runtime and memory notes

Input representation comes from Module A and is fixed before the hyperparameter search starts. It may differ from the deep model's input.

Imports `data.py` and `metrics.py`. Writes no split logic and no metric of its own.

### Module C: Deep model

- 1 model, depending on dataset modality (text, image, audio), among: MLP, CNN, RNN, or Convolutional autoencoder.
- Training loop, hyperparameters, overfitting controls
- Training and validation curves
- One pretrained variant if scope allows (ImageNet backbones such as ResNet50V2, InceptionV3, EfficientNetB0 for images or spectrograms, a pretrained audio or tabular encoder otherwise)

Same imports as Module B.

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
| G2 | Before 10/15 | Split and metrics frozen, `splits/split_v1.csv` committed and `verify_split.py` passing. B and C each have one model training on dummy or real data. |
| G3 | Before 10/31 | Every model produces the full metric set on the real split |
| G4 | Two weeks before final report | Results table populated, all figures drawn |

A gate is passed by showing running code.

---

## 6. Rules for consistency

- One repo, one branch per person, no code emailed around. `Project Overview.md` lives in the repo, since it is the document that defines everyone's interfaces.
- One frozen split, committed as `splits/split_v1.csv` and grouped on `lesion_id`, and one metrics function, both written before any model is trained.
- Every result reported with the same metric set, logged through `results.py`, and tagged with the `split_version` it was computed against.
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
  - `preprocess.py`: cleaning, conversion to model input, normalization, augmentation. Serves as the loader both model modules (B and C) import
- **Experimental:** Compare 2 to 3 preprocessing settings on one fixed model, produce the sample-inspection figure, document class distribution
- **Writing:** Dataset description, Preprocessing methods, data figure



## Member: Erick

- **Module B:** Classical baselines
- **Coding:** 
  - `baselines.py`: 3-4 baseline models
- **Experimental:** Hyperparameter search per model, k sweep for KNN, kernel comparison for SVM, runtime and memory measurement
- **Writing:** Baseline models methods, baseline experimental setup, baseline results analysis



## Member: Bryan

- **Module C:** Deep Learning model
- **Coding:** 
  - `model.py`: architecture, training loop, callbacks, checkpointing. 
  - `pretrained.py` for pretrained model if time permits
- **Experimental:** Learning rate and batch size sweep, regularization or dropout comparison, from-scratch versus pretrained comparison, training and validation curves
- **Writing:** Deep model architecture, training setup, deep model results analysis
