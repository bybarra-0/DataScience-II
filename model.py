import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
        return np.array([LABEL_TO_IDX[label] for label in y], dtype=np.int64)
    return np.array(y, dtype=np.int64)

def calculate_class_weights(y_indices: np.ndarray) -> torch.Tensor:
    # Counterbalance severe class imbalance during loss calculation
    present_classes = np.unique(y_indices)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_indices
    )
    weight_arr = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, w in zip(present_classes, weights):
        weight_arr[int(cls)] = float(w)
    return torch.tensor(weight_arr, dtype=torch.float32)

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

def plot_history(history: dict, model_name: str) -> None:
    # Plot training and validation loss and accuracy trajectories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("train_loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.get("train_acc", []), label="Train Acc")
    axes[1].plot(history.get("val_acc", []), label="Val Acc")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    output_path = FIGURES_DIR / f"history_{model_name.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

class CustomCNNStub(nn.Module):
    # Base template for Module C owner to define custom convolutional layers
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    # Interface stub: Module C owner defines and returns custom CNN here
    raise NotImplementedError("Module C owner will define the custom CNN architecture here.")

class _PipelineProbe(nn.Module):
    # Minimal network used exclusively to verify pipeline integration before architecture selection
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _build_pipeline_probe(num_classes: int = NUM_CLASSES) -> nn.Module:
    return _PipelineProbe(num_classes=num_classes)

def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    config_desc: str,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    use_class_weights: bool = True,
    class_weights: torch.Tensor = None,
    save_plots: bool = True,
    device: str = None
) -> dict:
    # Model-agnostic training and evaluation harness for any PyTorch module
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_device = torch.device(selected_device)
    model = model.to(target_device)

    # Accommodate legacy array-based positional calls during Phase 1 migration
    if isinstance(train_loader, np.ndarray):
        X_train, y_train = train_loader, val_loader
        train_loader, val_loader = None, None

    # Construct streaming loaders if in-memory arrays were supplied
    if train_loader is None:
        if X_train is None or y_train is None:
            raise ValueError("Either train_loader or (X_train, y_train) must be provided.")
        y_train_idx = encode_labels(y_train)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train_idx).long()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_loader is None:
        if X_val is None or y_val is None:
            raise ValueError("Either val_loader or (X_val, y_val) must be provided.")
        y_val_idx = encode_labels(y_val)
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val_idx).long()
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Derive or assign loss balancing weights
    if use_class_weights:
        if class_weights is not None:
            weights = class_weights.to(target_device)
        elif hasattr(train_loader.dataset, "tensors") and len(train_loader.dataset.tensors) > 1:
            targets_idx = train_loader.dataset.tensors[1].cpu().numpy()
            weights = calculate_class_weights(targets_idx).to(target_device)
        elif hasattr(train_loader.dataset, "targets"):
            targets = train_loader.dataset.targets
            targets_idx = encode_labels(targets if isinstance(targets, np.ndarray) else np.array(targets))
            weights = calculate_class_weights(targets_idx).to(target_device)
        else:
            weights = None
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    checkpoint_path = MODELS_DIR / f"{model_name.lower()}_best.pt"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_weights = None

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_train_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(target_device)
                labels = labels.to(target_device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

        # Check early stopping and checkpointing criteria
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    runtime_s = time.time() - start_time

    # Restore top weights before evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    all_val_preds = []
    all_val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(target_device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_targets.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_val_targets, all_val_preds)
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

    # Wrap synthetic data in DataLoaders to model Module A streaming contract
    y_train_idx = encode_labels(y_train)
    y_val_idx = encode_labels(y_val)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_idx).long()),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val_idx).long()),
        batch_size=16,
        shuffle=False
    )

    # 2. Module C owner creates their chosen model instance
    # Replace _build_pipeline_probe with build_model once custom architecture is chosen
    model = _build_pipeline_probe()

    # 3. Train and benchmark the selected architecture
    print("Testing pipeline execution with architecture-agnostic harness:")
    train_and_evaluate(
        model=model,
        model_name="PipelineProbe",
        config_desc="pipeline_sanity_check",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2
    )
