# =============================================================================
# config.py — Central configuration for all hyperparameters and paths
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = "/content/face-mask-detection"
DATA_DIR    = "/content/data/face-mask-dataset"
RAW_DIR     = os.path.join(DATA_DIR, "raw")          # original downloaded images
SPLIT_DIR   = os.path.join(DATA_DIR, "split")        # train / val / test split
TRAIN_DIR   = os.path.join(SPLIT_DIR, "train")
VAL_DIR     = os.path.join(SPLIT_DIR, "val")
TEST_DIR    = os.path.join(SPLIT_DIR, "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "omkargurav/face-mask-dataset"
CLASSES        = ["with_mask", "without_mask"]

# ---------------------------------------------------------------------------
# Split ratio  (must sum to 1.0)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE    = (224, 224)   # MobileNetV2 default input
IMG_SHAPE   = (224, 224, 3)
BATCH_SIZE  = 32

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS_FROZEN  = 10    # phase 1: train only top layers (backbone frozen)
EPOCHS_FINETUNE = 10   # phase 2: fine-tune last few backbone layers
LEARNING_RATE  = 1e-3
FINETUNE_LR    = 1e-5
UNFREEZE_LAYERS = 30   # number of layers to unfreeze from top of MobileNetV2

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
ROTATION_RANGE     = 20
WIDTH_SHIFT_RANGE  = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE         = 0.2
HORIZONTAL_FLIP    = True

# ---------------------------------------------------------------------------
# Model output paths
# ---------------------------------------------------------------------------
MODEL_BEST_PATH  = os.path.join(MODEL_DIR, "best_model.keras")
MODEL_FINAL_PATH = os.path.join(MODEL_DIR, "final_model.keras")
