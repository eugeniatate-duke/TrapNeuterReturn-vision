"""
Naive majority-class baseline for TNRVision.

The baseline always predicts the most common class
observed in the training dataset.
"""

from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd


TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")


CLASSES = [
    "cat",
    "dog",
    "raccoon",
    "opossum",
]


def count_training_examples():
    counts = {}

    for class_name in CLASSES:
        class_dir = TRAIN_DIR / class_name

        counts[class_name] = len(
            list(class_dir.glob("*"))
        )

    return counts


def load_test_labels():
    labels = []

    for class_name in CLASSES:
        class_dir = TEST_DIR / class_name

        for _ in class_dir.glob("*"):
            labels.append(class_name)

    return labels


def main():

    counts = count_training_examples()

    print("\nTraining Counts")
    print(counts)

    # majority_class = max(
    #     counts,
    #     key=counts.get,
    # )
    majority_class = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    )[0][0]

    print(
        f"\nMajority Class: {majority_class}"
    )

    y_true = load_test_labels()

    y_pred = [
        majority_class
        for _ in y_true
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    metrics_df = pd.DataFrame({
        "metric": [
            "accuracy",
            "macro_f1",
            "weighted_f1"

        ],

        "value": [
            accuracy,
            report["macro avg"]["f1-score"],
            report["weighted avg"]["f1-score"]
        ]
    })

    metrics_df.to_csv(
        "artifacts/baseline_metrics.csv",
        index=False,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES,
    )

    print("\nConfusion Matrix")
    print(cm)

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(
        "artifacts/baseline_confusion_matrix.csv"
    )


if __name__ == "__main__":
    main()