# Project Management and Course Coordination

Authoritative documents: [readme.md](file:///d:/F2026/DS2/Main%20Branch/readme.md) and [docs/Project Proposal.md](file:///d:/F2026/DS2/Main%20Branch/docs/Project%20Proposal.md)

---

## 1. Writing Map

Per proposal guidelines, group members share workload across experiments and writing.

### Final Report Writing Map

| Final Report Section | Primary Drafter |
|---|---|
| Abstract | Edward, written last |
| 1. Introduction | Shared |
| 2. Related Work | Shared |
| 3. Proposed Method: dataset and preprocessing | Module A owner (Edward) |
| 3. Proposed Method: baseline models | Module B owner (Erick) |
| 3. Proposed Method: deep model | Module C owner (Bryan) |
| 4. Experiments | Each member writes own setup |
| 5. Results and Discussion | Each member analyzes own models from results assembled by Edward |
| 6. Conclusions | Edward |
| 7. Team Contributions | Written together, updated at conclusion |

Edward sets up the final report document skeleton, headings, figure placeholders, citation style, and performs final editorial pass for consistency. Each member drafts into that skeleton.

---

## 2. Checkpoints

| Gate | Target Date | Requirement |
|---|---|---|
| G1 | Before proposal due on Sep 18 | Dataset downloaded and loading, one sample batch inspected |
| G2 | Before 10/15 | Split and metrics frozen, [splits/splits.csv](file:///d:/F2026/DS2/Main%20Branch/splits/splits.csv) committed. Module B and C each have one model training on dummy or real data |
| G3 | Before 10/31 | Every model produces full metric set on real split |
| G4 | Two weeks before final report | Results table populated, all figures drawn |

A gate is passed by showing running code.

---

## 3. Rules for Consistency

- One repo, one branch per person, no code emailed around. [readme.md](file:///d:/F2026/DS2/Main%20Branch/readme.md) lives in the repo root to define module interface contracts.
- One frozen split grouped on `lesion_id`, committed as [splits/splits.csv](file:///d:/F2026/DS2/Main%20Branch/splits/splits.csv).
- One shared metrics evaluation function [`compute_metrics`](file:///d:/F2026/DS2/Main%20Branch/metrics.py#L10), written before model training.
- Every result logged to [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) with identical metric schema.
- Model selection occurs strictly on the validation split. The held-out test split is evaluated once at G3.

---

## 4. Course Deliverables

Project accounts for 30 percent of the final course grade.

| Deliverable | Specification |
|---|---|
| Proposal | 1 to 3 pages, 800 to 1200 words, PDF |
| Presentation | Recorded video and slides, 8 to 10 min plus 2 min Q and A |
| Written Report | 6 to 8 pages excluding references |
| Poster | Digital and printed presentation format |

Reproducibility standard: The report must be detailed enough that a classmate could replicate results without guesswork.

Grading focus: Grades reflect learning process, execution, and honest reporting rather than raw benchmark scores. Analysis of failure cases and future improvements is valued equally with positive performance.

---

## 5. Team Contributions Draft

### Member: Edward

- Module: Shared Infrastructure and Module A (Preprocessing)
- Coding:
  - [config.py](file:///d:/F2026/DS2/Main%20Branch/config.py)
  - [data.py](file:///d:/F2026/DS2/Main%20Branch/data.py) (ingestion, split generation, verification routines)
  - [metrics.py](file:///d:/F2026/DS2/Main%20Branch/metrics.py)
  - [requirements.txt](file:///d:/F2026/DS2/Main%20Branch/requirements.txt)
  - [results.csv](file:///d:/F2026/DS2/Main%20Branch/results.csv) schema
  - [preprocess.py](file:///d:/F2026/DS2/Main%20Branch/preprocess.py) (metadata cleaning, age imputation, 64-component PCA pipeline, PyTorch [`HAM10000Dataset`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L245) and [`get_deep_loader`](file:///d:/F2026/DS2/Main%20Branch/preprocess.py#L268), train-only augmentations)
- Experimental: Split balance verification, leakage check, seed stability, pipeline sanity run, resize strategy benchmark comparing 3 image geometries on a fixed Random Forest baseline, generating [figures/sample_inspection.png](file:///d:/F2026/DS2/Main%20Branch/figures/sample_inspection.png) and [figures/class_distribution.png](file:///d:/F2026/DS2/Main%20Branch/figures/class_distribution.png)
- Writing: Abstract, Introduction, Related Work, Evaluation Protocol, Dataset and Preprocessing methodology, resize benchmark analysis, Conclusions, final editorial pass

### Member: Erick

- Module: Module B (Classical Baselines)
- Coding: [baselines.py](file:///d:/F2026/DS2/Main%20Branch/baselines.py) (Logistic Regression, KNN, SVM, Random Forest)
- Experimental: Hyperparameter search per model, k sweep for KNN, kernel comparison for SVM, runtime and memory measurement
- Writing: Baseline models methodology, baseline experimental setup, baseline results analysis

### Member: Bryan

- Module: Module C (Deep Learning Models)
- Coding: [model.py](file:///d:/F2026/DS2/Main%20Branch/model.py) (custom CNN), [pretrained.py](file:///d:/F2026/DS2/Main%20Branch/pretrained.py) (pretrained EfficientNetB0)
- Experimental: Learning rate and batch size sweep, regularization and dropout comparison, from-scratch versus transfer learning comparison, training and validation loss curves
- Writing: Deep model architecture, training setup, deep model results analysis
