# =============================================================================
# prepare_data.py — Download dataset from Kaggle and split into train/val/test
# =============================================================================

import os
import sys
import shutil
import zipfile
import random
from pathlib import Path
from tqdm import tqdm

# Allow running from project root or src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    KAGGLE_DATASET, RAW_DIR, SPLIT_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, TRAIN_RATIO, VAL_RATIO, DATA_DIR
)

SEED = 42


def download_dataset() -> None:
    """Download the Kaggle dataset and extract it to RAW_DIR."""
    os.makedirs(RAW_DIR, exist_ok=True)

    zip_name = KAGGLE_DATASET.split("/")[-1] + ".zip"
    zip_path = os.path.join(DATA_DIR, zip_name)

    print("⬇️  Downloading dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --quiet")

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    os.remove(zip_path)
    print(f"✅ Dataset successfully extracted to: {RAW_DIR}")


def find_class_dirs(base: str, classes: list) -> dict:
    """
    Search recursively for directories matching class names.
    Returns dict: {class_name: Path}
    """
    found = {}
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            if d in classes and d not in found:
                found[d] = os.path.join(root, d)
    return found


def split_dataset() -> None:
    """Split raw images into train / val / test directories."""
    random.seed(SEED)

    # Locate class directories inside the extracted folder
    class_dirs = find_class_dirs(RAW_DIR, CLASSES)

    if len(class_dirs) != len(CLASSES):
        missing = set(CLASSES) - set(class_dirs.keys())
        raise FileNotFoundError(
            f"Class directory not found: {missing}\n"
            f"Detected structure in {RAW_DIR}:"
        )

    # Create split directories
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

    print("\n✂️  Splitting the dataset...")

    for cls, src_dir in class_dirs.items():
        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        random.shuffle(images)

        n          = len(images)
        n_train    = int(n * TRAIN_RATIO)
        n_val      = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train : n_train + n_val],
            "test" : images[n_train + n_val :],
        }

        for split_name, files in splits.items():
            dst_dir = os.path.join(SPLIT_DIR, split_name, cls)
            for fname in tqdm(files, desc=f"  {cls}/{split_name}", leave=False):
                shutil.copy(os.path.join(src_dir, fname),
                            os.path.join(dst_dir, fname))

        print(f"  📂 {cls}: {len(splits['train'])} train | "
              f"{len(splits['val'])} val | {len(splits['test'])} test")


def verify_split() -> None:
    """Print final count of images per split per class."""
    print("\n📊 Verify dataset distribution:")
    print(f"  {'Split':<8} {'with_mask':>12} {'without_mask':>14} {'Total':>8}")
    print("  " + "-" * 46)

    grand_total = 0
    for split in ["train", "val", "test"]:
        counts = {}
        for cls in CLASSES:
            path = os.path.join(SPLIT_DIR, split, cls)
            counts[cls] = len(os.listdir(path)) if os.path.exists(path) else 0
        total = sum(counts.values())
        grand_total += total
        print(f"  {split:<8} {counts['with_mask']:>12} "
              f"{counts['without_mask']:>14} {total:>8}")

    print("  " + "-" * 46)
    print(f"  {'TOTAL':<8} {'':>12} {'':>14} {grand_total:>8}")


if __name__ == "__main__":
    download_dataset()
    split_dataset()
    verify_split()
    print("\n🎉 Data preparation complete! Ready for training.")
