import numpy as np
from config import IMAGE_SIZE, CLASS_NAMES

def transform(image_path: str) -> np.ndarray:
    # Interface signature: converts an image file into a normalized (224, 224, 3) array
    # To be replaced by Module A with real Pillow or OpenCV reading and scaling logic
    raise NotImplementedError("Real image transformation will be implemented by Module A.")

def dummy_batch(batch_size: int = 32, modality: str = "deep"):
    # Generates synthetic data so Modules B and C can build pipelines immediately
    y = np.random.choice(CLASS_NAMES, size=batch_size)

    if modality == "deep":
        # Simulates a batch of RGB image tensors normalized to [0, 1]
        X = np.random.rand(batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype(np.float32)
        return X, y
    elif modality == "classical":
        # Simulates flattened tabular feature vectors (e.g. 64 extracted features)
        n_features = 64
        X = np.random.randn(batch_size, n_features).astype(np.float32)
        return X, y
    else:
        valid_modalities = ["deep", "classical"]
        raise ValueError(f"modality must be one of {valid_modalities}, received: {modality}")

def get_classical_data(split_name: str = "train"):
    # Interface signature: loads feature matrix X and label vector y for Module B
    # Module A will replace this with real dataset feature extraction
    raise NotImplementedError("Real feature extraction will be implemented by Module A.")

if __name__ == "__main__":
    # Sanity check for downstream developers
    X_cnn, y_cnn = dummy_batch(batch_size=8, modality="deep")
    print("Deep batch verification:", X_cnn.shape, y_cnn.shape)

    X_base, y_base = dummy_batch(batch_size=8, modality="classical")
    print("Classical batch verification:", X_base.shape, y_base.shape)