from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
TEST_DIR = "data/processed/test"

ARTIFACT_DIR = Path("artifacts/deep")
MODEL_DIR = Path("models")

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        4,
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    print("Training...")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {running_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "models/mobilenet_v3_small.pth",
    )

    print("Evaluating...")

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
            "value": [
                accuracy,
                report["macro avg"]["f1-score"],
                report["weighted avg"]["f1-score"],
            ],
        }
    ).to_csv(
        ARTIFACT_DIR / "metrics.csv",
        index=False,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    pd.DataFrame(cm).to_csv(
        ARTIFACT_DIR / "confusion_matrix.csv",
        index=False,
    )


if __name__ == "__main__":
    main()