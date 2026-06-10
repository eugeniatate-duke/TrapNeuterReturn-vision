"""
Create train/validation/test splits for TNRVision.

Author: Eugenia Tate
"""

from pathlib import Path
import random
import shutil

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CLASSES = [
    "cat",
    "dog",
    "raccoon",
    "opossum",
]


def create_directory_structure():
    """Create train/val/test directories."""

    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            (PROCESSED_DIR / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )


def get_images(class_name):
    """Return all jpg/jpeg/png files."""

    class_dir = RAW_DIR / class_name

    images = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(class_dir.rglob(extension))

    return images


def split_images(images):
    """Split images into train/val/test."""

    random.shuffle(images)

    n = len(images)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_images(images, split_name, class_name):
    """Copy images to processed directory."""

    destination = PROCESSED_DIR / split_name / class_name

    for image_path in images:
        shutil.copy2(
            image_path,
            destination / image_path.name,
        )


def main():

    random.seed(RANDOM_SEED)

    create_directory_structure()

    print("\nDataset Summary")
    print("-" * 40)

    for class_name in CLASSES:

        images = get_images(class_name)

        train, val, test = split_images(images)

        copy_images(train, "train", class_name)
        copy_images(val, "val", class_name)
        copy_images(test, "test", class_name)

        print(
            f"{class_name:<10}"
            f"total={len(images):<5}"
            f"train={len(train):<5}"
            f"val={len(val):<5}"
            f"test={len(test):<5}"
        )


if __name__ == "__main__":
    main()