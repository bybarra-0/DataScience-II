import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import SEED, NUM_CLASSES, seed_everything
from preprocess import get_deep_loader
from model import train_and_evaluate


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    # TODO: Instantiate torchvision.models.efficientnet_b0 with pretrained ImageNet weights
    # If freeze_backbone is True, freeze feature parameters to train only the classifier head
    # Replace final linear layer: model.classifier[1] = nn.Linear(in_features, num_classes)
    raise NotImplementedError("TODO: Implement EfficientNetB0 transfer model in pretrained.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module C: EfficientNetB0 Transfer Learning")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze convolutional feature extractor")
    args = parser.parse_args()

    seed_everything(SEED)
    print("Module C: Loading PyTorch DataLoaders from Module A for Transfer Learning...")
    train_loader = get_deep_loader(split_name="train", batch_size=args.batch_size, shuffle=True)
    val_loader = get_deep_loader(split_name="val", batch_size=args.batch_size, shuffle=False)

    # TODO: Build model using build_efficientnet_b0 and invoke train_and_evaluate()