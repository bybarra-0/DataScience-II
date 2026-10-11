import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from config import SEED, CLASS_NAMES, RESULTS_PATH, PROJECT_ROOT
from preprocess import get_classical_data
from metrics import compute_metrics


def log_baseline_result(model_name: str, config_desc: str, metrics: dict, runtime_s: float) -> dict:
    # Maintain identical reporting schema across modules in central tracking file
    log_row = {
        "model_name": model_name,
        "module": "B",
        "config": config_desc,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "runtime_s": round(runtime_s, 2)
    }

    # Append row directly to shared csv without overwriting prior records
    row_df = pd.DataFrame([log_row])
    row_df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    return log_row


def evaluate_and_log(model, model_name: str, config_desc: str, X_val: np.ndarray, y_val: np.ndarray, runtime_s: float) -> dict:
    # Measure generalization on held-out validation split
    preds = model.predict(X_val)
    metrics = compute_metrics(y_val, preds)
    return log_baseline_result(model_name, config_desc, metrics, runtime_s)


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Wrap LogisticRegression in StandardScaler pipeline, set class_weight="balanced", and fit on X_train
    raise NotImplementedError("TODO: Implement Logistic Regression pipeline in baselines.py")


def train_knn(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Wrap KNeighborsClassifier in StandardScaler pipeline and fit on X_train
    raise NotImplementedError("TODO: Implement KNN pipeline in baselines.py")


def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Wrap SVC in StandardScaler pipeline, configure kernel and class_weight="balanced", and fit on X_train
    raise NotImplementedError("TODO: Implement SVM pipeline in baselines.py")


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Initialize RandomForestClassifier with class_weight="balanced" and random_state=SEED, and fit on X_train
    raise NotImplementedError("TODO: Implement Random Forest in baselines.py")


def tune_model(estimator, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray, cv_splits: int = 5):
    # TODO: Configure StratifiedKFold and GridSearchCV optimizing balanced_accuracy scoring
    raise NotImplementedError("TODO: Implement hyperparameter tuning in baselines.py")


def export_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path = None):
    # TODO: Plot confusion matrix heatmap using seaborn and save to figures/
    raise NotImplementedError("TODO: Implement confusion matrix export in baselines.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module B: Classical Machine Learning Baselines")
    parser.add_argument("--model", type=str, choices=["lr", "knn", "svm", "rf", "all"], default="all", help="Model architecture to train")
    parser.add_argument("--tune", action="store_true", help="Execute hyperparameter grid search")
    args = parser.parse_args()

    print("Module B: Loading classical feature arrays from Module A...")
    # Ingest 64-component PCA projections and string target labels
    X_train, y_train = get_classical_data(split_name="train")
    X_val, y_val = get_classical_data(split_name="val")
    print(f"Loaded train features: {X_train.shape}, validation features: {X_val.shape}")

    # TODO: Execute baseline training and hyperparameter search routines