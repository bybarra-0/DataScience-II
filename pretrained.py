import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as tv_models

from config import SEED, NUM_CLASSES, seed_everything
from preprocess import get_deep_loader
from model import train_and_evaluate, evaluation_checkpoint


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    #a function that hands us a predefined model for us to compare
    if pretrained:
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 # weights for the images that we pick. 
    else:
        weights = None
    model = tv_models.efficientnet_b0(weights = weights) #defines model with the weight
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False # causes the head to learn
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module C: EfficientNetB0 Transfer Learning")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze convolutional feature extractor")
    parser.add_argument("--eval-split", type = str, default= None, choices= ["val", "test"])
    args = parser.parse_args()

    seed_everything(SEED)

    model_text = "frozen" if args.freeze_backbone else "unfrozed"
    config_desc = "backbone_" + model_text + "_lr" + str(args.lr) + "_batchSize" + str(args.batch_size)

    if args.eval_split is not None:
        eval_loader = get_deep_loader(split_name=args.eval_split, batch_size= args.batch_size, shuffle= False)
        model = build_efficientnet_b0(num_classes= NUM_CLASSES, pretrained= False, freeze_backbone= args.freeze_backbone)
        evaluation_checkpoint(model, "efficientnet_b0", config_desc, eval_loader, args.eval_split)
        raise SystemExit


    print("Module C: Loading PyTorch DataLoaders from Module A for Transfer Learning...")
    train_loader = get_deep_loader(split_name="train", batch_size=args.batch_size, shuffle=True)
    val_loader = get_deep_loader(split_name="val", batch_size=args.batch_size, shuffle=False)

    model = build_efficientnet_b0(
        num_classes= NUM_CLASSES,
        pretrained= True,
        freeze_backbone= args.freeze_backbone
    )

    summary = train_and_evaluate(
        model= model,
        model_name= "efficientnet_b0",
        config_desc= config_desc, 
        train_loader= train_loader,
        val_loader= val_loader,
        epochs= args.epochs,
        lr=args.lr
    )

    print(summary) # essentially a overview