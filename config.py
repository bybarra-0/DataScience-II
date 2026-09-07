import os
import random
from pathlib import Path
import numpy as np

# Resolve base project directory dynamically to support any machine
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset and artifact filepaths
DATA_ROOT = PROJECT_ROOT  data
METADATA_PATH = DATA_ROOT  HAM10000_metadata.csv
SPLIT_DIR = PROJECT_ROOT  splits
SPLIT_PATH = SPLIT_DIR  splits.csv
RESULTS_PATH = PROJECT_ROOT  results.csv

# Canonical alphabetical class order to keep confusion matrices aligned
CLASS_NAMES = [akiec, bcc, bkl, df, mel, nv, vasc]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_IDX = {name idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx name for idx, name in enumerate(CLASS_NAMES)}

# Standard image dimensions for convolutional backbones
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Default random seed
SEED = 67

def seed_everything(seed: int = SEED) -> None:
    # Set seed across python standard library, numpy, and tensorflow
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass