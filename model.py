import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    CLASS_NAMES,
    INPUT_SHAPE,
    LABEL_TO_IDX,
    NUM_CLASSES,
    PROJECT_ROOT,
    RESULTS_PATH,
    SEED,
    seed_everything,
)
from metrics import compute_metrics
from preprocess import dummy_batch

FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "deep"

def encode_labels(y: np.ndarray) -> np.ndarray:
    # Convert string class names to integer indices aligned with CLASS_NAMES
    if len(y) > 0 and isinstance(y[0], str):
        return np.array([LABEL_TO_IDX[label] for label in y], dtype=np.int32)
    return np.array(y, dtype=np.int32)

def calculate_class_weights(y_indices: np.ndarray) -> dict:
    # Counterbalance class imbalance while handling missing classes during dummy testing
    present_classes = np.unique(y_indices)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_indices
    )
    weight_dict = {cls: 1.0 for cls in range(NUM_CLASSES)}
    weight_dict.update({int(cls): float(w) for cls, w in zip(present_classes, weights)})
    return weight_dict

def log_result(model_name: str, config_desc: str, metrics: dict, runtime_s: float) -> None:
    # Append model run to shared results.csv under module C
    row = {
        "model_name": model_name,
        "module": "C",
        "config": config_desc,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "runtime_s": round(runtime_s, 2)
    }
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)

def plot_confusion_matrix(cm: list, model_name: str) -> None:
    # Save confusion matrix heatmap to figures directory
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

def plot_history(history: keras.callbacks.History, model_name: str) -> None:
    # Plot training and validation loss and accuracy trajectories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history.get("loss", []), label="Train Loss")
    axes[0].plot(history.history.get("val_loss", []), label="Val Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history.get("accuracy", []), label="Train Acc")
    axes[1].plot(history.history.get("val_accuracy", []), label="Val Acc")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    output_path = FIGURES_DIR / f"history_{model_name.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

def build_model(input_shape: tuple = INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> keras.Model:
    # Interface stub: Module C owner defines and compiles their chosen architecture here
    # (e.g. custom CNN, transfer learning backbone, autoencoder, or MLP)
    raise NotImplementedError("Module C owner will define the model architecture here.")

def _build_pipeline_probe(input_shape: tuple = INPUT_SHAPE, num_classes: int = NUM_CLASSES) -> keras.Model:
    # Minimal linear model used exclusively to verify pipeline integration before architecture selection
    inputs = keras.Input(shape=input_shape)
    x = layers.GlobalAveragePooling2D()(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    probe = keras.Model(inputs=inputs, outputs=outputs, name="PipelineProbe")
    probe.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return probe

def train_and_evaluate(
    model: keras.Model,
    model_name: str,
    config_desc: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    use_class_weights: bool = True,
    save_plots: bool = True
) -> dict:
    # Model-agnostic training and evaluation harness for any Keras architecture
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y_train_idx = encode_labels(y_train)
    y_val_idx = encode_labels(y_val)

    class_weights = calculate_class_weights(y_train_idx) if use_class_weights else None

    checkpoint_path = MODELS_DIR / f"{model_name.lower()}_best.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train_idx,
        validation_data=(X_val, y_val_idx),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    runtime_s = time.time() - start_time

    val_probs = model.predict(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    metrics = compute_metrics(y_val_idx, val_preds)
    log_result(model_name, config_desc, metrics, runtime_s)

    if save_plots:
        plot_confusion_matrix(metrics["confusion_matrix"], model_name)
        plot_history(history, model_name)

    print(f"[{model_name}] {runtime_s:.2f}s | Bal Acc: {metrics['balanced_accuracy']} | Macro F1: {metrics['macro_f1']}")
    return metrics

if __name__ == "__main__":
    seed_everything(SEED)

    # 1. Load synthetic data to test the pipeline immediately
    X_train, y_train = dummy_batch(batch_size=64, modality="deep")
    X_val, y_val = dummy_batch(batch_size=32, modality="deep")

    # 2. Module C owner creates their chosen model instance
    # Replace _build_pipeline_probe with build_model once implemented
    model = _build_pipeline_probe()

    # 3. Train and benchmark the selected architecture
    print("Testing pipeline execution with architecture-agnostic harness:")
    train_and_evaluate(
        model=model,
        model_name="PipelineProbe",
        config_desc="pipeline_sanity_check",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=2,
        batch_size=16
    )
