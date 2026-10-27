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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
    raise NotImplementedError("TODO: Implement Logistic Regression pipeline in baselines.py")


def train_knn(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Wrap KNeighborsClassifier in StandardScaler pipeline and fit on X_train
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
    raise NotImplementedError("TODO: Implement KNN pipeline in baselines.py")


def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Wrap SVC in StandardScaler pipeline, configure kernel and class_weight="balanced", and fit on X_train
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(class_weight="balanced", random_state=SEED))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
    
    raise NotImplementedError("TODO: Implement SVM pipeline in baselines.py")


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    # TODO: Initialize RandomForestClassifier with class_weight="balanced" and random_state=SEED, and fit on X_train
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED)
    rf.fit(X_train, y_train)
    return rf
    raise NotImplementedError("TODO: Implement Random Forest in baselines.py")


def tune_model(estimator, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray, cv_splits: int = 5):
    # TODO: Configure StratifiedKFold and GridSearchCV optimizing balanced_accuracy scoring
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
    raise NotImplementedError("TODO: Implement hyperparameter tuning in baselines.py")


def export_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path = None):
    # TODO: Plot confusion matrix heatmap using seaborn and save to figures/
    if output_path is None:
        output_dir = PROJECT_ROOT / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"

    labels = CLASS_NAMES if CLASS_NAMES is not None else np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {output_path}")

   

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
    models_to_run = ["lr", "knn", "svm", "rf"] if args.model == "all" else [args.model]
    param_grids = {
        "lr": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "saga"]
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 7, 11, 15, 21, 25],
            "classifier__weights": ["uniform", "distance"]
        },
        "svm": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf"]
        },
        "rf": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    }

    model_registry = {
        "lr": ("Logistic Regression", train_logistic_regression),
        "knn": ("KNN", train_knn),
        "svm": ("SVM", train_svm),
        "rf": ("Random Forest", train_random_forest)
    }

    for model_key in models_to_run:
        display_name, build_fn = model_registry[model_key]
        print(f"\n==========================================")
        print(f" Running {display_name} (Tuning: {args.tune})")
        print(f"==========================================")

        start_time = time.time()

        if args.tune:
            # Instantiate pipeline/estimator skeleton to pass into GridSearchCV
            base_model = build_fn(X_train, y_train)  # Dummy fit to generate estimator instance
            fitted_model, best_params = tune_model(base_model, param_grids[model_key], X_train, y_train)
            config_desc = f"tuned_{best_params}"
        else:
            fitted_model = build_fn(X_train, y_train)
            config_desc = "default"

        elapsed_time = time.time() - start_time

        # Record validation performance in results.csv
        logged_record = evaluate_and_log(
            model=fitted_model,
            model_name=display_name,
            config_desc=config_desc,
            X_val=X_val,
            y_val=y_val,
            runtime_s=elapsed_time
        )
        print(f"Results logged: {logged_record}")

        # Export confusion matrix heatmap
        val_predictions = fitted_model.predict(X_val)
        matrix_name = f"{display_name}_{'tuned' if args.tune else 'default'}"
        export_confusion_matrix(y_val, val_predictions, matrix_name)