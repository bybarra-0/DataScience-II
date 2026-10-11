import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import SEED, NUM_CLASSES, INPUT_SHAPE, RESULTS_PATH, PROJECT_ROOT, seed_everything
from data import get_class_weights
from preprocess import get_deep_loader
from metrics import compute_metrics


def log_deep_result(model_name: str, config_desc: str, metrics: dict, runtime_s: float) -> dict:
    # Maintain identical reporting schema across modules in central tracking file
    log_row = {
        "model_name": model_name,
        "module": "C",
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


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # TODO: Define convolutional feature extractor and fully connected classification head
        # Input tensor geometry: (batch_size, 3, 224, 224)
        # Expected output shape: (batch_size, num_classes)
        raise NotImplementedError("TODO: Define CustomCNN layers in model.py")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass returning raw unnormalized logits
        raise NotImplementedError("TODO: Implement CustomCNN forward pass in model.py")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    # TODO: Iterate over mini-batches, execute forward pass, backpropagation, and optimizer step
    raise NotImplementedError("TODO: Implement train_one_epoch loop in model.py")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict]:
    # TODO: Evaluate model in torch.no_grad() mode
    # Convert raw logits to integer class predictions via torch.argmax(logits, dim=1).cpu().numpy()
    # Evaluate performance by passing integer targets and predictions to compute_metrics(y_true, y_pred)
    raise NotImplementedError("TODO: Implement evaluate loop in model.py")


def plot_training_curves(train_losses: list, val_losses: list, val_accs: list, output_path: Path = None):
    # TODO: Export training and validation loss and accuracy curves to figures/
    raise NotImplementedError("TODO: Implement plot_training_curves in model.py")


def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    config_desc: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = None
) -> dict:
    # TODO: complete training execution
    # 1. Counter class imbalance: criterion = nn.CrossEntropyLoss(weight=get_class_weights(as_tensor=True).to(device))
    # 2. Configure optimizer, learning rate scheduler, and early stopping
    # 3. Track best validation balanced accuracy and save checkpoint to models/deep/{model_name}_{config_desc}_best.pt
    # 4. Export training curves using plot_training_curves()
    # 5. Log final run metrics to results.csv using log_deep_result()
    raise NotImplementedError("TODO: Implement train_and_evaluate orchestration in model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module C: Deep Learning Model Training")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    seed_everything(SEED)
    print("Module C: Initializing PyTorch DataLoaders from Module A...")
    train_loader = get_deep_loader(split_name="train", batch_size=args.batch_size, shuffle=True)
    val_loader = get_deep_loader(split_name="val", batch_size=args.batch_size, shuffle=False)
    print(f"DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # TODO: Instantiate CustomCNN and execute train_and_evaluate()