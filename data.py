import argparse
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config

data_root = getattr(config, "data_root".upper())
metadata_path = getattr(config, "metadata_path".upper())
split_dir = getattr(config, "split_dir".upper())
split_path = getattr(config, "split_path".upper())
class_names = getattr(config, "class_names".upper())
seed = getattr(config, "seed".upper())

def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    # Confirm metadata classes match canonical class order
    assert set(df["dx"].unique()) == set(class_names), "Classes in metadata do not match canonical class list"
    return df

def generate_split() -> pd.DataFrame:
    df = load_metadata()

    # Deduplicate by lesion id to prevent identical lesions straddling splits
    unique_lesions = df.drop_duplicates(subset="lesion_id").copy()

    # Partition 70 percent train and 30 percent holdout
    train_lesions, temp_lesions = train_test_split(
        unique_lesions,
        test_size=0.30,
        random_state=seed,
        stratify=unique_lesions["dx"]
    )

    # Partition holdout evenly into 15 percent validation and 15 percent test
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=0.50,
        random_state=seed,
        stratify=temp_lesions["dx"]
    )

    # Map split membership back to all image entries
    lesion_to_split = {
        **{lid: "train" for lid in train_lesions["lesion_id"]},
        **{lid: "val" for lid in val_lesions["lesion_id"]},
        **{lid: "test" for lid in test_lesions["lesion_id"]}
    }
    df["split"] = df["lesion_id"].map(lesion_to_split)

    # Assert zero lesion leakage across partition boundaries
    train_ids = set(df[df["split"] == "train"]["lesion_id"])
    val_ids = set(df[df["split"] == "val"]["lesion_id"])
    test_ids = set(df[df["split"] == "test"]["lesion_id"])
    assert len(train_ids & val_ids) == 0, "Leakage detected between train and val"
    assert len(train_ids & test_ids) == 0, "Leakage detected between train and test"
    assert len(val_ids & test_ids) == 0, "Leakage detected between val and test"

    # Persist split manifest to disk
    split_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_path, index=False)
    return df

def get_split(split: str = None, split_name: str = None) -> pd.DataFrame:
    # Read frozen split manifest to ensure deterministic data access across modules
    target_split = split if split is not None else split_name
    if not split_path.exists():
        generate_split()
    df = pd.read_csv(split_path)
    if target_split is not None:
        valid_splits = ["train", "val", "test"]
        if target_split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}")
        return df[df["split"] == target_split].reset_index(drop=True)
    return df

def verify_images(auto_extract: bool = True) -> bool:
    # Validate that all 10,015 image files exist and are readable on disk
    df = load_metadata()
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = set(df["image_id"].astype(str))
    existing_images = {p.stem for p in images_dir.glob("*.jpg") if p.stat().st_size > 0}

    # Automatically extract Harvard Dataverse zip parts if images directory is incomplete
    if len(existing_images) < len(expected_ids) and auto_extract:
        dataset_prefix = "ham10000".upper()
        zip_names = [f"{dataset_prefix}_images_part_1.zip", f"{dataset_prefix}_images_part_2.zip"]
        for zip_name in zip_names:
            zip_file = data_root / zip_name
            if zip_file.exists():
                with zipfile.ZipFile(zip_file, "r") as archive:
                    archive.extractall(images_dir)
        existing_images = {p.stem for p in images_dir.glob("*.jpg") if p.stat().st_size > 0}

    missing = expected_ids - existing_images
    if missing:
        print(f"Missing or corrupted images: {len(missing)} of {len(expected_ids)}")
        return False
    return True

def get_class_counts(split: str = "train", split_name: str = None) -> dict:
    # Compute class sample distribution in canonical alphabetical order
    target_split = split if split_name is None else split_name
    df = get_split(target_split)
    counts = df["dx"].value_counts().to_dict()
    return {cls: int(counts.get(cls, 0)) for cls in class_names}

def get_class_weights(as_tensor: bool = False):
    # Compute inverse-frequency class weights strictly on the training partition
    counts = get_class_counts("train")
    n_train = sum(counts.values())
    k = len(class_names)
    weights = {cls: float(n_train / (k * counts[cls])) for cls in class_names}

    if as_tensor:
        # Deep learning loss functions require a float tensor matching class index order
        import torch
        ordered_weights = [weights[cls] for cls in class_names]
        return torch.tensor(ordered_weights, dtype=torch.float32)

    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset management and verification")
    parser.add_argument("--verify", action="store_true", help="Validate image integrity on disk")
    args = parser.parse_args()

    if args.verify:
        is_valid = verify_images(auto_extract=True)
        print("Image verification passed:", is_valid)
    else:
        split_df = generate_split()
        print("Split generated successfully at:", split_path)
        print(split_df["split"].value_counts())