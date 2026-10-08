import os
import random
from pathlib import Path
import numpy as np

# Resolve base project directory dynamically to support any machine
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset and artifact filepaths
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_PATH = DATA_ROOT / "HAM10000_metadata.csv"
SPLIT_DIR = PROJECT_ROOT / "splits"
SPLIT_PATH = SPLIT_DIR / "splits.csv"
RESULTS_PATH = PROJECT_ROOT / "results.csv"

# Canonical alphabetical class order to keep confusion matrices aligned
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Standard image dimensions for convolutional backbones
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (3, 224, 224)

# Default random seed
SEED = 67

def seed_everything(seed: int = SEED) -> None:
    # Set seed across python runtime, numpy, and pytorch for reproducible experiments
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
