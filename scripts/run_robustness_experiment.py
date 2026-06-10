from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from PIL import ImageFilter

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

TEST_DIR = "data/processed/test"

ARTIFACT_DIR = Path(
    "artifacts/experiments"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class GaussianBlurTransform:
    def __call__(self, image):
        return image.filter(
            ImageFilter.GaussianBlur(radius=2)
        )


def evaluate(transform):

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
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

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    return accuracy_score(
        y_true,
        y_pred,
    )


def main():

    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dark_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(
            brightness=0.3
        ),
        transforms.ToTensor(),
    ])

    blur_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        GaussianBlurTransform(),
        transforms.ToTensor(),
    ])

    results = []

    experiments = {
        "original": original_transform,
        "dark": dark_transform,
        "blur": blur_transform,
    }

    for name, transform in experiments.items():

        print(
            f"Evaluating {name}..."
        )

        accuracy = evaluate(
            transform
        )

        print(
            f"{name}: {accuracy:.4f}"
        )

        results.append(
            {
                "condition": name,
                "accuracy": accuracy,
            }
        )

    pd.DataFrame(
        results
    ).to_csv(
        ARTIFACT_DIR /
        "robustness_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()