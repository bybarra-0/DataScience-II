import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, SEED, seed_everything
from model import train_and_evaluate
from preprocess import dummy_batch

def build_backbone(pretrained: bool = True) -> nn.Module:
    # Instantiates the EfficientNetB0 backbone specified in the project proposal
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    return models.efficientnet_b0(weights=weights)

def _build_probe_backbone() -> nn.Module:
    # Lightweight uninitialized backbone for immediate pipeline verification without weight downloads
    return models.efficientnet_b0(weights=None)

def build_transfer_model(
    base_model: nn.Module,
    num_classes: int = NUM_CLASSES,
    dense_units: int = 128,
    dropout_rate: float = 0.3
) -> nn.Module:
    # Freeze feature extractor parameters to preserve pretrained weights during head warmup
    for param in base_model.parameters():
        param.requires_grad = False

    # Replace original ImageNet classification head with custom multi-class head
    in_features = base_model.classifier[1].in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, dense_units),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(dense_units, num_classes)
    )
    return base_model

def unfreeze_for_fine_tuning(
    model: nn.Module,
    unfreeze_from_stage: int = 6
) -> nn.Module:
    # Selectively unfreeze top convolutional stages for end-to-end fine-tuning
    if hasattr(model, "features"):
        for stage_idx, stage in enumerate(model.features):
            if stage_idx >= unfreeze_from_stage:
                for param in stage.parameters():
                    param.requires_grad = True
    return model

if __name__ == "__main__":
    seed_everything(SEED)

    # 1. Load synthetic data to verify transfer learning flow in channels-first format
    X_train, y_train = dummy_batch(batch_size=64, modality="deep")
    X_val, y_val = dummy_batch(batch_size=32, modality="deep")

    # 2. Instantiate backbone (probe used here to avoid blocking on weight downloads)
    backbone = _build_probe_backbone()

    # 3. Stage 1: Train classification head with frozen backbone
    transfer_model = build_transfer_model(
        base_model=backbone,
        dense_units=128,
        dropout_rate=0.3
    )

    print("Testing Stage 1: Feature extraction with frozen backbone:")
    train_and_evaluate(
        model=transfer_model,
        model_name="Pretrained_Stage1_Probe",
        config_desc="frozen_backbone_head_dense128",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=2,
        batch_size=16,
        learning_rate=1e-4
    )

    # 4. Stage 2: Fine-tune top layers with reduced learning rate
    print("\nTesting Stage 2: Fine-tuning top convolutional stages:")
    fine_tuned_model = unfreeze_for_fine_tuning(
        model=transfer_model,
        unfreeze_from_stage=6
    )

    train_and_evaluate(
        model=fine_tuned_model,
        model_name="Pretrained_Stage2_Probe",
        config_desc="fine_tuned_top_stages_lr1e-5",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=1,
        batch_size=16,
        learning_rate=1e-5
    )
