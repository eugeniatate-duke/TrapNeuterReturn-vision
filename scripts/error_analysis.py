from pathlib import Path
import shutil

import pandas as pd
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

TEST_DIR = "data/processed/test"

ERROR_DIR = Path(
    "artifacts/errors"
)

ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASS_NAMES = [
    "cat",
    "dog",
    "opossum",
    "raccoon",
]


def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    model = models.mobilenet_v3_small()

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        4,
    )

    model.load_state_dict(
        torch.load(
            "models/mobilenet_v3_small.pth",
            map_location=DEVICE,
        )
    )

    model = model.to(DEVICE)
    model.eval()

    rows = []

    count = 0

    for idx in range(len(dataset)):

        image, label = dataset[idx]

        with torch.no_grad():

            prediction = model(
                image.unsqueeze(0).to(DEVICE)
            )

            predicted_class = (
                prediction.argmax(dim=1)
                .item()
            )

        if predicted_class != label:

            image_path = dataset.samples[idx][0]

            destination = (
                ERROR_DIR /
                f"error_{count+1}.jpg"
            )

            shutil.copy2(
                image_path,
                destination,
            )

            rows.append(
                {
                    "image":
                        destination.name,
                    "actual":
                        CLASS_NAMES[label],
                    "predicted":
                        CLASS_NAMES[
                            predicted_class
                        ],
                }
            )

            count += 1

            if count >= 5:
                break

    pd.DataFrame(rows).to_csv(
        ERROR_DIR /
        "error_analysis.csv",
        index=False,
    )

    print(
        f"Saved {count} errors."
    )


if __name__ == "__main__":
    main()