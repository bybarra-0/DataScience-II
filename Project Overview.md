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
Headline metric (set from class distribution): Balanced Multiclass Accuracy (mean per-class sensitivity / macro recall)
```

---

## 2. Shared infrastructure

| Item | Output |
|---|---|
| Frozen train / validation / test split | `data.py`, fixed seed, identical results on every call |
| Shared metrics function | `metrics.py`, one fixed metric set for every model |
| Results table | `results.py`, one table, all models, same metric set |

`data.py` exposes one function taking a fixed seed and an optional grouping key. If the source publishes an official split, we adopt it and cite it instead of generating our own. If it does not, we generate a stratified split and, where a grouping key exists, group on it so no ID appears in two splits. The choice and the reason are recorded in Section 2 before any model is trained.

`metrics.py` returns accuracy, macro precision, macro recall, macro F1, per-class recall, and the confusion matrix. All six are always computed. Which one leads the results table is set once, from the class distribution recorded in Section 2: accuracy if the classes are within roughly 3 to 1, macro F1 plus per-class recall otherwise.

**Rationale:** if three people split the data independently, the model comparison is compromised and the report cannot be defended. Everything depends on this first.

**One of us needs to own the split function, the seed, and the interface signature.** I (Edward) will take this block. It is small and everything depends on it, so it gets done once and first.

- The preprocessing owner (Module A) writes the transforms and calls into that interface. Neither edits the other's file.

---

## 3. Open modules (pick one each)

**Dependency:** Shared Infrastructure -> Module A -> Module B + C

### Module A: Preprocessing and data loading

- Dataset acquisition, cleaning, and any parsing of annotation files
- Conversion to a fixed-shape model input (images: resize and normalize. audio: segment, length-normalize, convert to spectrogram or extract features. tabular: encode and scale)
- Augmentation appropriate to the modality
- Class balance handling and documentation of what was done
- Sample inspection, at least one figure showing what the data looks like
- Agrees with Module B on the feature representation classical models will consume, since it is usually not the same tensor Module C takes

Module A implements `preprocess.transform()`. Modules B and C build against the interface stub using dummy data initially, then swap in real data when A finishes.

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
| G2 | Before 10/15 | Split and metrics frozen. B and C each have one model training on dummy or real data. |
| G3 | Before 10/31 | Every model produces the full metric set on the real split |
| G4 | Two weeks before final report | Results table populated, all figures drawn |

A gate is passed by showing running code.

---

## 6. Rules for consistency

- One repo, one branch per person, no code emailed around.
- One frozen split, grouped on the declared key if one exists, and one metrics function, both written before any model is trained.
- Every result reported with the same metric set.

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



## Member: Edward N

- **Module:** Shared Infrastructure

- **Coding:** `data.py`, `metrics.py`, `results.py`

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
