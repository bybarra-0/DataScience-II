<style>
  body {
    font-size: 10.5pt;
    line-height: 1.35;
  }
  h1 {
    font-size: 15pt;
    margin-top: 0;
    margin-bottom: 6px;
  }
  h2 {
    font-size: 12.5pt;
    margin-top: 10px;
    margin-bottom: 4px;
  }
  p {
    margin-top: 0;
    margin-bottom: 6px;
  }
  ul {
    margin-top: 2px;
    margin-bottom: 6px;
    padding-left: 20px;
  }
  li {
    margin-bottom: 3px;
  }
</style>

# COSC 4337 Group 6 Project Proposal: Multi-Class Skin Lesion Classification
**Team Members:** Edward Nguyen, Erick Saldana, Bryan Ybarra

## Introduction

Automated medical image analysis plays an increasingly critical role in healthcare. Early detection of high-risk skin cancers relies heavily on dermatoscopic examinations, yet manual visual inspection exhibits high diagnostic variability depending on clinician experience. Consequently, automated skin lesion classification has become a central focus in medical artificial intelligence. Foundational work showed deep convolutional neural networks matching board-certified dermatologists on benchmark datasets [[1]](#ref1), establishing a precedent expanded by international initiatives like the ISIC challenges [[2]](#ref2).

This project benchmarks classical machine learning against deep learning architectures for classifying skin lesions across seven diagnostic categories. We evaluate classical baselines (Logistic Regression, Support Vector Machines, K-Nearest Neighbors, and Random Forests) against a custom convolutional neural network and a fine-tuned EfficientNetB0 backbone, quantifying trade-offs between classification performance and computational overhead.

Experiments utilize the HAM10000 dataset [[2]](#ref2) from the Harvard Dataverse, containing 10015 dermatoscopic images across 7470 unique lesions. The dataset spans benign conditions such as melanocytic nevi and dermatofibromas to malignant neoplasms including melanoma and basal cell carcinoma. To prevent data leakage, multiple images of the same lesion are grouped strictly by lesion ID, ensuring models learn generalizable features despite an extreme 58 to 1 class imbalance.

## Motivation

Applying machine learning directly to clinical healthcare challenges is our primary motivation. Skin cancer causes thousands of deaths globally each month, yet early detection yields exceptionally high patient survival rates. However, routine visual examinations suffer from high false-negative rates, and annual screening rates remain low, particularly among underserved minority populations. We are motivated to develop an automated screening pipeline that acts as a clinical safety net, prioritizing high sensitivity on malignant lesions to assist clinicians and extend diagnostic coverage.

Technically, this project provides direct experience solving real-world medical data engineering bottlenecks. Our priority is establishing leak-free dataset partitioning, artifact reduction, and balanced preprocessing to handle severe class imbalance. Furthermore, benchmarking classical algorithms against deep convolutional architectures enables our team to evaluate practical trade-offs between predictive performance, computational efficiency, and clinical deployability in medical AI.

## Evaluation

Success is defined by establishing a leak-free evaluation pipeline and training models that significantly outperform naive baselines on the imbalanced HAM10000 dataset. Because the majority class (melanocytic nevi) comprises roughly 67 percent of the dataset, an uninformative majority-class classifier achieves 67 percent raw accuracy. Therefore, technical success cannot be measured by raw accuracy alone.

We measure success across three technical criteria:

- Baseline Outperformance: Both classical baselines (Logistic Regression, Random Forest, Support Vector Machines, and K-Nearest Neighbors) and convolutional neural networks must exceed majority-class prediction. Deep models must demonstrate clear performance gains over classical feature baselines.
- Imbalance-Aware Metrics: The primary headline metric is balanced multiclass accuracy (macro recall), which averages sensitivity across all seven categories. Macro F1-score, macro precision, and confusion matrices track class-wide trade-offs. Achieving balanced accuracy above 70 percent on the test set indicates technical success.
- Malignant Class Sensitivity: False negatives on malignant lesions are unacceptable clinically. Success requires per-class recall for melanoma (mel) and basal cell carcinoma (bcc) to exceed 75 percent.

All evaluations will be conducted on a static 15 percent held-out test split grouped on lesion_id to eliminate patient-level data leakage.

## Resources

Dataset: Experiments utilize the HAM10000 dataset, comprising 10015 dermatoscopic images (7470 unique lesions, roughly 3.0 GB) and diagnostic metadata hosted on the Harvard Dataverse.

Computational Tools: The project is implemented in Python. Shared preprocessing relies on Pandas for metadata indexing, NumPy for array operations, Pillow for image resizing (224x224 pixels), and Scikit-learn for stratified splitting. Classical baselines utilize Scikit-learn to train Logistic Regression, KNN, SVM, and Random Forests with StandardScaler and GridSearchCV. Deep learning models (custom CNN and fine-tuned EfficientNetB0) are implemented in PyTorch. Git tracks versioned split manifests and metric logs.

Hardware and Compute: Data preprocessing and classical model searches execute on local multi-core CPU workstations. Deep neural network training utilizes cloud-based GPU runtimes through Google Colab to accelerate training and validation.

## Team Contributions

Workload is distributed evenly across all team members, with each person responsible for distinct computational, experimental, and writing tasks.

- Edward Nguyen (Shared Infrastructure and Preprocessing): Manages repository architecture, data ingestion pipelines, static split generation, and metric tracking. Implements image preprocessing, resizing to 224x224, normalization, and augmentations. Validates zero lesion leakage across splits and produces exploratory data figures. Drafts dataset descriptions, preprocessing methodology, evaluation protocols, and coordinates document editing.
- Erick Saldana (Classical Baselines): Implements classical baselines (Logistic Regression, KNN, SVM, Random Forests) via Scikit-learn with StandardScaler. Conducts hyperparameter tuning via GridSearchCV with StratifiedKFold cross-validation, benchmarks execution runtimes, and logs baseline performance metrics. Drafts baseline modeling methodologies, optimization setups, and comparative baseline results analysis.
- Bryan Ybarra (Deep Learning): Builds a custom convolutional neural network and transfer learning pipeline using EfficientNetB0 in PyTorch. Executes GPU training loops in Google Colab, tunes hyperparameters, tracks loss curves, and applies regularization controls. Drafts deep learning architecture descriptions, training specifications, loss curve evaluations, and deep model results analysis.
- Shared Responsibilities: All members collaboratively conduct error analysis using confusion matrices, prepare presentation slides and recorded video, and assemble the project poster.

## References

<span id="ref1">[1]</span> A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," Nature, vol. 542, no. 7639, pp. 115-118, 2017.

<span id="ref2">[2]</span> P. Tschandl, C. Rosendahl, and H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," Scientific Data, vol. 5, no. 180161, 2018. doi: 10.1038/sdata.2018.161. Harvard Dataverse: https://doi.org/10.7910/DVN/DBW86T