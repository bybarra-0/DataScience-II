import pandas as pd
from sklearn.model_selection import train_test_split
from config import METADATA_PATH, SPLIT_PATH, SPLIT_DIR, SEED, CLASS_NAMES

def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_PATH)
    # Validate expected class labels
    assert set(df["dx"].unique()) == set(CLASS_NAMES), "Classes in metadata do not match CLASS_NAMES"
    return df

def generate_split() -> pd.DataFrame:
    df = load_metadata()

    # Deduplicate by lesion_id to prevent multiple photos of one lesion crossing splits
    unique_lesions = df.drop_duplicates(subset="lesion_id").copy()

    # Stage 1: 70% train, 30% temp
    train_lesions, temp_lesions = train_test_split(
        unique_lesions,
        test_size=0.30,
        random_state=SEED,
        stratify=unique_lesions["dx"]
    )

    # Stage 2: 15% validation, 15% test
    val_lesions, test_lesions = train_test_split(
        temp_lesions,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_lesions["dx"]
    )

    # Map split tags back to all 10,015 images
    lesion_to_split = {
        **{lid: "train" for lid in train_lesions["lesion_id"]},
        **{lid: "val" for lid in val_lesions["lesion_id"]},
        **{lid: "test" for lid in test_lesions["lesion_id"]}
    }
    df["split"] = df["lesion_id"].map(lesion_to_split)

    # Assert zero patient or lesion leakage across any split boundary
    train_ids = set(df[df["split"] == "train"]["lesion_id"])
    val_ids = set(df[df["split"] == "val"]["lesion_id"])
    test_ids = set(df[df["split"] == "test"]["lesion_id"])
    assert len(train_ids & val_ids) == 0, "Leakage detected between train and val"
    assert len(train_ids & test_ids) == 0, "Leakage detected between train and test"
    assert len(val_ids & test_ids) == 0, "Leakage detected between val and test"

    # Save to disk as static artifact
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLIT_PATH, index=False)
    return df

def get_split(split_name: str = None) -> pd.DataFrame:
    if not SPLIT_PATH.exists():
        generate_split()
    df = pd.read_csv(SPLIT_PATH)
    if split_name is not None:
        valid_splits = ["train", "val", "test"]
        if split_name not in valid_splits:
            raise ValueError(f"split_name must be one of {valid_splits}")
        return df[df["split"] == split_name].reset_index(drop=True)
    return df

if __name__ == "__main__":
    split_df = generate_split()
    print("Split generated successfully at:", SPLIT_PATH)
    print(split_df["split"].value_counts())