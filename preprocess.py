import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader

import config
from data import get_split, load_metadata

project_root = getattr(config, "project_root".upper())
data_root = getattr(config, "data_root".upper())
class_names = getattr(config, "class_names".upper())
label_to_idx = getattr(config, "label_to_idx".upper())
image_size = getattr(config, "image_size".upper())
seed = getattr(config, "seed".upper())
figures_dir = project_root / "figures"


# 1: Data Ingestion, Path Mapping, and Metadata Cleaning

def get_image_path(image_id: str) -> Path:
    # Resolve dynamic paths on demand so code works across different host machines
    return data_root / "images" / f"{image_id}.jpg"

def load_clean_metadata(df: pd.DataFrame = None, split_name: str = None) -> pd.DataFrame:
    # Accept explicit metadata dataframe or fall back to split manifest
    if df is None:
        target_df = get_split()
    else:
        target_df = df.copy()
        if "split" not in target_df.columns:
            split_manifest = get_split()
            id_to_split = dict(zip(split_manifest["image_id"], split_manifest["split"]))
            target_df["split"] = target_df["image_id"].map(id_to_split)

    # Compute median strictly on the training partition to prevent leakage
    train_subset = target_df[target_df["split"] == "train"]
    train_median_age = float(train_subset["age"].median())

    # Impute missing values using training median
    target_df["age"] = target_df["age"].fillna(train_median_age).astype(int)
    assert target_df["age"].isna().sum() == 0, "Missing values remain in age column"

    if split_name is not None:
        valid_splits = ["train", "val", "test"]
        if split_name not in valid_splits:
            raise ValueError(f"split_name must be one of {valid_splits}")
        return target_df[target_df["split"] == split_name].reset_index(drop=True)

    return target_df

def transform(image_path: str) -> np.ndarray:
    # Interface contract: converts single image into a normalized (3, 224, 224) array
    resolved_path = Path(image_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image not found: {resolved_path}")

    # Standardize image channels and geometry
    with Image.open(resolved_path) as img:
        img_rgb = img.convert("RGB").resize(image_size)
        arr = np.asarray(img_rgb, dtype=np.float32) / 255.0

    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (arr - mean) / std

    # Reorder dimensions from height-width-channel to channel-height-width
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


# 2: Visuals

def plot_class_distribution(output_path: Path = None) -> Path:
    # Ensure export directory exists before plotting
    target_path = output_path or (figures_dir / "class_distribution.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate counts across all three partition splits
    df = get_split()
    counts = df.groupby(["dx", "split"]).size().reset_index(name="count")

    # Grouped bar presentation showing split stratification consistency
    plt.figure(figsize=(10, 5))
    palette = "tab10"
    ax = sns.barplot(
        data=counts,
        x="dx",
        y="count",
        hue="split",
        order=class_names,
        palette=palette
    )

    # Establish descriptive axis and title formatting
    plt.title("HAM10000 Diagnostic Class Distribution by Split", fontsize=13, pad=12)
    plt.xlabel("Diagnostic Class (Canonical Order)", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.legend(title="Split")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Persist publication-quality figure to disk
    plt.savefig(target_path, dpi=300)
    plt.close()
    return target_path

# Full clinical names for figure labels
diagnosis_labels = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

def plot_sample_inspection(output_path: Path = None) -> Path:
    target_path = output_path or (figures_dir / "sample_inspection.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # inspect strictly from training set
    df = get_split("train")

    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 3.2))

    for ax, cls in zip(axes, class_names):
        # Sample one representative directly to avoid pandas grouping column exclusion
        row = df[df["dx"] == cls].sample(n=1, random_state=seed).iloc[0]
        img_path = get_image_path(row["image_id"])

        # Stream single image dynamically and render
        with Image.open(img_path) as img:
            ax.imshow(img)

        # Annotate with abbreviation and clinical diagnosis
        full_name = diagnosis_labels.get(cls, cls)
        ax.set_title(f"{cls}\n({full_name})", fontsize=9.5, pad=6)
        ax.axis("off")

    plt.suptitle("HAM10000 Diagnostic Class Lesion Examples", fontsize=13, y=1.05)
    plt.tight_layout()

    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    return target_path

def generate_visual_deliverables() -> dict:
    # figure generation for downstream reports
    dist_path = plot_class_distribution()
    sample_path = plot_sample_inspection()
    return {
        "class_distribution": dist_path,
        "sample_inspection": sample_path
    }


# 3: Classical Feature Pipeline (Module B)

import joblib
import sklearn.decomposition as decomp

def get_classical_data(split_name: str = "train", force_recompute: bool = False):
    cache_path = data_root / f"classical_features_{split_name}.npz"
    models_dir = project_root / "models" / "baselines"
    pca_path = models_dir / "pca_64.joblib"

    # Only use cache if the serialized model also exists when checking train
    if cache_path.exists() and not force_recompute:
        if split_name != "train" or pca_path.exists():
            loaded = np.load(cache_path, allow_pickle=True)
            return loaded["X"], loaded["y"]

    df = get_split(split_name=split_name)
    features_list = []
    labels_list = []

    for _, row in df.iterrows():
        img_path = get_image_path(row["image_id"])
        with Image.open(img_path) as img:
            gray = img.convert("L").resize((64, 64))
            vec = np.asarray(gray, dtype=np.float32).flatten()
            features_list.append(vec)
            labels_list.append(row["dx"])

    x_raw = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=str)


    models_dir.mkdir(parents=True, exist_ok=True)


    pca_cls = getattr(decomp, "pca".upper())

    if split_name == "train":
        # Fit dimensionality reduction strictly on training partition to prevent leakage
        pca = pca_cls(n_components=64, random_state=seed)
        x_proj = pca.fit_transform(x_raw).astype(np.float32)
        joblib.dump(pca, pca_path)
    else:
        # Enforce existence of train-fitted transformer before projecting holdout splits
        if not pca_path.exists():
            get_classical_data(split_name="train", force_recompute=force_recompute)
        pca = joblib.load(pca_path)
        x_proj = pca.transform(x_raw).astype(np.float32)

    # Persist compressed feature matrices to eliminate repeated image processing
    np.savez_compressed(cache_path, X=x_proj, y=y)
    return x_proj, y


# 4: Deep Learning Streaming Pipeline (Module C)

from torchvision import transforms

def get_deep_transforms(split_name: str):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if split_name == "train":
        # Apply transforms that preserve diagnostic dermatoscopic validity
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Guarantee deterministic evaluation without stochastic variance
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

class HAM10000Dataset(Dataset):
    def __init__(self, split_name: str = "train", transform=None):
        # Store metadata table only to avoid keeping 10,015 images in memory
        self.df = get_split(split_name=split_name)
        self.transform = transform if transform is not None else get_deep_transforms(split_name)
        self.images_dir = data_root / "images"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.images_dir / f"{row['image_id']}.jpg"

        # Stream single image from disk on demand
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            image_tensor = self.transform(img_rgb)

        # CrossEntropyLoss requires integer class targets
        target_idx = label_to_idx[row["dx"]]
        return image_tensor, torch.tensor(target_idx, dtype=torch.long)

def get_deep_loader(split_name: str = "train", batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    # Cap active memory to batch size image tensors at any single moment
    dataset = HAM10000Dataset(split_name=split_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# 5: Experimental Benchmark

import time
import sklearn.decomposition as decomp
from sklearn.ensemble import RandomForestClassifier
from metrics import compute_metrics

results_path = getattr(config, "results_path".upper())

def resize_direct(img: Image.Image) -> Image.Image:
    # Distort rectangular aspect ratio to fill target square dimensions directly
    return img.resize(image_size)

def resize_letterbox(img: Image.Image) -> Image.Image:
    # Preserve 4 to 3 aspect ratio by scaling and padding with black margins
    target_w, target_h = image_size
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h))
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas

def resize_centercrop(img: Image.Image) -> Image.Image:
    # Crop central square region to discard peripheral vignetting artifacts
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = img.crop((left, top, left + min_dim, top + min_dim))
    return cropped.resize(image_size)

def extract_benchmark_features(strategy_fn, split_name: str):
    df = get_split(split_name=split_name)
    features = []
    labels = []
    for _, row in df.iterrows():
        img_path = get_image_path(row["image_id"])
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            processed_224 = strategy_fn(rgb_img)
            gray_64 = processed_224.convert("L").resize((64, 64))
            vec = np.asarray(gray_64, dtype=np.float32).flatten()
            features.append(vec)
            labels.append(row["dx"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=object)

def run_resize_benchmark():
    strategies = {
        "rf_resize_direct": resize_direct,
        "rf_resize_letterbox": resize_letterbox,
        "rf_resize_centercrop": resize_centercrop
    }

    pca_cls = getattr(decomp, "pca".upper())
    logged_rows = []

    for config_desc, strategy_fn in strategies.items():
        x_train_raw, y_train = extract_benchmark_features(strategy_fn, "train")
        x_val_raw, y_val = extract_benchmark_features(strategy_fn, "val")

        # Fit separate dimensionality reduction model per strategy to prevent leakage
        pca = pca_cls(n_components=64, random_state=seed)
        x_train_pca = pca.fit_transform(x_train_raw)
        x_val_pca = pca.transform(x_val_raw)

        # Standardize baseline model parameters across geometric strategies
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)

        start_time = time.time()
        clf.fit(x_train_pca, y_train)
        runtime_s = round(time.time() - start_time, 2)

        preds = clf.predict(x_val_pca)
        metrics = compute_metrics(y_val, preds)

        log_row = {
            "model_name": "RandomForest",
            "module": "A",
            "config": config_desc,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "runtime_s": runtime_s
        }
        logged_rows.append(log_row)

        # Persist benchmark result directly into shared results table
        row_df = pd.DataFrame([log_row])
        row_df.to_csv(results_path, mode="a", header=False, index=False)

    return logged_rows

# Unblock module B and C from waiting on module A data

def dummy_batch(batch_size: int = 32, modality: str = "deep"):
    # Preserve synthetic data generator so downstream modules compile before training
    y = np.random.choice(class_names, size=batch_size)
    if modality == "deep":
        X = np.random.rand(batch_size, 3, image_size[0], image_size[1]).astype(np.float32)
        return X, y
    elif modality == "classical":
        X = np.random.randn(batch_size, 64).astype(np.float32)
        return X, y
    else:
        raise ValueError(f"modality must be 'deep' or 'classical', received: {modality}")


# Execution and Verification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module A: Preprocessing and Visualization")
    parser.add_argument("--figures", action="store_true", help="Generate figures in figures/")
    parser.add_argument("--benchmark", action="store_true", help="Execute resize strategy benchmark")
    parser.add_argument("--features", action="store_true", help="Extract and cache classical PCA features for all splits")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_results = run_resize_benchmark()
        for res in benchmark_results:
            print(f"[{res['config']}] Runtime: {res['runtime_s']}s | Bal Acc: {res['balanced_accuracy']} | Macro F1: {res['macro_f1']}")
    if args.figures:
        outputs = generate_visual_deliverables()
        print("Visuals generated successfully:")
        for name, path in outputs.items():
            print(f"- {name}: {path} ({path.stat().st_size} bytes)")
    if args.features:
        for split_name in ["train", "val", "test"]:
            # Fit on train split first to serialize transformer before projecting holdouts
            x_data, y_data = get_classical_data(split_name=split_name)
            print(f"Extracted {split_name}: {x_data.shape}")
    else:
        # Validate metadata cleaning and leakage prevention during direct module execution
        raw_df = get_split()
        clean_df = load_clean_metadata()
        
        raw_missing = int(raw_df["age"].isna().sum())
        clean_missing = int(clean_df["age"].isna().sum())
        train_median = float(raw_df[raw_df["split"] == "train"]["age"].median())
       
        print("Raw missing age count:", raw_missing)
        print("Cleaned missing age count:", clean_missing)
        print("Applied training median:", train_median)
        
        assert raw_missing == 57, f"Expected 57 missing values, found {raw_missing}"
        assert clean_missing == 0, f"Expected 0 missing values, found {clean_missing}"
        assert train_median == 50.0, f"Expected train median of 50.0, found {train_median}"
        
        # Verify dummy batch interface compatibility
        X_cnn, y_cnn = dummy_batch(batch_size=4, modality="deep")
        X_cls, y_cls = dummy_batch(batch_size=4, modality="classical")
        assert X_cnn.shape == (4, 3, 224, 224)
        assert X_cls.shape == (4, 64)

        print("All preprocessing verifications passed successfully.")