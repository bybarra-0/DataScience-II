import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASS_NAMES, PROJECT_ROOT, RESULTS_PATH, SEED
from metrics import compute_metrics
from preprocess import dummy_batch

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "baselines"

def build_pipeline(estimator, scale: bool = True):
    # Conditionally prepend standard scaling while preserving consistent pipeline step naming
    if scale:
        return make_pipeline(StandardScaler(), estimator)
    return make_pipeline(estimator)

def log_result(model_name: str, config_desc: str, metrics: dict, runtime_s: float) -> None:
    # Append any model run to shared results.csv
    row = {
        "model_name": model_name,
        "module": "B",
        "config": config_desc,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "runtime_s": round(runtime_s, 2)
    }
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)

def plot_confusion_matrix(cm: list, model_name: str) -> None:
    # Export confusion matrix figure for report inclusion
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    output_path = FIGURES_DIR / f"cm_{model_name.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_model(estimator, model_name: str) -> None:
    # Serialize fitted model or pipeline to models/baselines/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(estimator, MODELS_DIR / f"{model_name.lower()}.joblib")

def evaluate_model(estimator, name: str, config_desc: str, X_train, y_train, X_val, y_val, save_plot: bool = True) -> dict:
    # Fit any scikit-learn estimator, record wall-clock time, and compute metrics
    start = time.time()
    estimator.fit(X_train, y_train)
    runtime = time.time() - start

    preds = estimator.predict(X_val)
    metrics = compute_metrics(y_val, preds)

    log_result(name, config_desc, metrics, runtime)
    save_model(estimator, name)
    if save_plot:
        plot_confusion_matrix(metrics["confusion_matrix"], name)

    print(f"[{name}] {runtime:.2f}s | Bal Acc: {metrics['balanced_accuracy']} | Macro F1: {metrics['macro_f1']}")
    return metrics

def tune_model(estimator, param_grid: dict, name: str, X_train, y_train, X_val, y_val, cv_splits: int = 3):
    # Universal tuning harness supporting any estimator and parameter search grid
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    start = time.time()
    grid.fit(X_train, y_train)
    runtime = time.time() - start

    best_model = grid.best_estimator_
    best_preds = best_model.predict(X_val)
    metrics = compute_metrics(y_val, best_preds)

    config_desc = f"tuned_{grid.best_params_}"
    log_result(f"{name}_Tuned", config_desc, metrics, runtime)
    save_model(best_model, f"{name}_Tuned")
    plot_confusion_matrix(metrics["confusion_matrix"], f"{name}_Tuned")

    print(f"[{name}_Tuned] {runtime:.2f}s | Best params: {grid.best_params_} | Bal Acc: {metrics['balanced_accuracy']}")
    return best_model

if __name__ == "__main__":
    # 1. Load data (swap dummy_batch for real get_classical_data when Module A finishes)
    X_train, y_train = dummy_batch(batch_size=200, modality="classical")
    X_val, y_val = dummy_batch(batch_size=50, modality="classical")

    # 2. Define any dictionary of candidate classical estimators
    # Below are examples for DecisionTree, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    candidate_models = {
        "DecisionTree": build_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            scale=False
        ),
        "LogisticRegression": build_pipeline(
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED),
            scale=True
        )
    }

    # 3. Evaluate all untuned baselines
    print("Evaluating candidate baseline models:")
    for name, model in candidate_models.items():
        evaluate_model(model, name, "default", X_train, y_train, X_val, y_val)

    # 4. Tune any selected model using an arbitrary parameter grid
    print("\nRunning hyperparameter tuning:")
    dt_param_grid = {
        "decisiontreeclassifier__max_depth": [3, 5, 10, None],
        "decisiontreeclassifier__min_samples_split": [2, 5]
    }
    tune_model(candidate_models["DecisionTree"], dt_param_grid, "DecisionTree", X_train, y_train, X_val, y_val)