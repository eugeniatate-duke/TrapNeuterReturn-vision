"""
Extract HOG features for TNRVision.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog


DATA_DIR = Path("data/processed/train")

CLASSES = [
    "cat",
    "dog",
    "raccoon",
    "opossum",
]


def extract_features(image_path):

    image = Image.open(image_path)

    image = image.convert("L")

    image = image.resize((128, 128))

    image = np.array(image)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    return features


def main():

    rows = []

    for label in CLASSES:

        class_dir = DATA_DIR / label

        for image_path in class_dir.glob("*"):

            try:

                features = extract_features(image_path)

                row = features.tolist()

                row.append(label)

                rows.append(row)

            except Exception as e:

                print(
                    f"Skipping {image_path}: {e}"
                )

    df = pd.DataFrame(rows)

    output_path = (
        "artifacts/hog_train_features.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(df)} samples"
    )


if __name__ == "__main__":
    main()