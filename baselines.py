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

    new_row_df = pd.DataFrame([log_row])
    if RESULTS_PATH.exists():
        results_df = pd.read_csv(RESULTS_PATH)
        match_idx = results_df.index[
            (results_df["model_name"] == model_name) &
            (results_df["module"] == "B") &
            (results_df["config"] == config_desc)
        ]
        # Update existing record in place to prevent duplicate rows on re-runs
        if not match_idx.empty:
            for col, val in log_row.items():
                results_df.loc[match_idx[0], col] = val
        else:
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        try:
            results_df.to_csv(RESULTS_PATH, index=False)
        except PermissionError:
            fallback_path = RESULTS_PATH.with_name("results_latest.csv")
            results_df.to_csv(fallback_path, index=False)
            print(f"Warning: {RESULTS_PATH.name} is locked by an external program. Wrote metrics to {fallback_path.name}")
    else:
        new_row_df.to_csv(RESULTS_PATH, index=False)

    return log_row


def evaluate_and_log(model, model_name: str, config_desc: str, X_eval: np.ndarray, y_eval: np.ndarray, runtime_s: float) -> tuple[dict, np.ndarray]:
    # Measure generalization on held-out evaluation split (val or test)
    preds = model.predict(X_eval)
    metrics = compute_metrics(y_eval, preds)
    logged_row = log_baseline_result(model_name, config_desc, metrics, runtime_s)
    return logged_row, preds


def train_logistic_regression(X_train: np.ndarray = None, y_train: np.ndarray = None):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=2000, tol=1e-2))
    ])
    if X_train is not None and y_train is not None:
        pipeline.fit(X_train, y_train)
    return pipeline


def train_knn(X_train: np.ndarray = None, y_train: np.ndarray = None):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier())
    ])
    if X_train is not None and y_train is not None:
        pipeline.fit(X_train, y_train)
    return pipeline


def train_svm(X_train: np.ndarray = None, y_train: np.ndarray = None):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(class_weight="balanced", random_state=SEED))
    ])
    if X_train is not None and y_train is not None:
        pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train: np.ndarray = None, y_train: np.ndarray = None):
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED)
    if X_train is not None and y_train is not None:
        rf.fit(X_train, y_train)
    return rf


def tune_model(estimator, param_grid: dict, X_train: np.ndarray, y_train: np.ndarray, cv_splits: int = 5):
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


def is_best_model(model_name: str, balanced_accuracy: float, split_name: str = "val") -> bool:
    # Check central results log to determine if this configuration achieves peak balanced accuracy
    if not RESULTS_PATH.exists():
        return True
    try:
        results_df = pd.read_csv(RESULTS_PATH)
        history = results_df[(results_df["model_name"] == model_name) & (results_df["module"] == "B")]
        if split_name == "test":
            history = history[history["config"].astype(str).str.startswith("test_")]
        else:
            history = history[~history["config"].astype(str).str.startswith("test_")]
        if history.empty:
            return True
        return balanced_accuracy >= history["balanced_accuracy"].max()
    except Exception:
        return True


def export_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path = None, title_suffix: str = ""):
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
    title_text = f"Confusion Matrix - {model_name}"
    if title_suffix:
        title_text += f" ({title_suffix})"
    plt.title(title_text)
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
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Evaluation partition (val or test)")
    parser.add_argument("--eval-test", action="store_true", help="Shortcut to evaluate on the frozen test partition")
    args = parser.parse_args()

    eval_split = "test" if args.eval_test else args.split

    print("Module B: Loading classical feature arrays from Module A...")
    # Ingest 64-component PCA projections and string target labels
    X_train, y_train = get_classical_data(split_name="train")
    X_eval, y_eval = get_classical_data(split_name=eval_split)
    print(f"Loaded train features: {X_train.shape}, {eval_split} features: {X_eval.shape}")

    models_to_run = ["lr", "knn", "svm", "rf"] if args.model == "all" else [args.model]
    param_grids = {
        "lr": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["saga"]
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
        print(f" Running {display_name} (Tuning: {args.tune}, Split: {eval_split})")
        print(f"==========================================")

        start_time = time.time()

        if args.tune:
            # Pass unfitted estimator to avoid redundant initial fitting
            base_model = build_fn()
            fitted_model, best_params = tune_model(base_model, param_grids[model_key], X_train, y_train)
            base_config = f"tuned_{best_params}"
        else:
            fitted_model = build_fn(X_train, y_train)
            base_config = "default"

        config_desc = f"test_{base_config}" if eval_split == "test" else base_config
        elapsed_time = time.time() - start_time

        # Record evaluation performance in results.csv and retrieve precomputed predictions
        logged_record, eval_predictions = evaluate_and_log(
            model=fitted_model,
            model_name=display_name,
            config_desc=config_desc,
            X_eval=X_eval,
            y_eval=y_eval,
            runtime_s=elapsed_time
        )
        print(f"Results logged: {logged_record}")

        # Export configuration-specific heatmap
        config_suffix = "tuned" if args.tune else "default"
        matrix_suffix = f"{eval_split}_{config_suffix}" if eval_split == "test" else config_suffix
        title_tag = f"Test - {config_suffix.capitalize()}" if eval_split == "test" else config_suffix.capitalize()
        export_confusion_matrix(
            y_eval,
            eval_predictions,
            model_name=f"{display_name}_{matrix_suffix}",
            title_suffix=title_tag
        )

        # Automatically export canonical heatmap for the best model per architecture
        if is_best_model(display_name, logged_record["balanced_accuracy"], split_name=eval_split):
            canonical_name = f"{display_name}_test" if eval_split == "test" else display_name
            canonical_title = f"Test Best - {config_suffix.capitalize()}" if eval_split == "test" else f"Best - {config_suffix.capitalize()}"
            export_confusion_matrix(
                y_eval,
                eval_predictions,
                model_name=canonical_name,
                title_suffix=canonical_title
            )
