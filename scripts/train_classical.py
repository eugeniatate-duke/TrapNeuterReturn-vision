"""
Train HOG + SVM classifier.
"""
import os
import pandas as pd
import joblib

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import (
    train_test_split,
)


def main():

    df = pd.read_csv(
        "artifacts/hog_train_features.csv"
    )

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    model = LinearSVC(
        random_state=42,
        max_iter=5000,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_val
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[
            "cat",
            "dog",
            "raccoon",
            "opossum",
        ],
    )

    os.makedirs("artifacts/classical", exist_ok=True)

    pd.DataFrame(
        cm,
        index=[
            "cat",
            "dog",
            "raccoon",
            "opossum",
        ],
        columns=[
            "cat",
            "dog",
            "raccoon",
            "opossum",
        ],
    ).to_csv(
        "artifacts/classical/confusion_matrix.csv"
    )

    accuracy = accuracy_score(
        y_val,
        predictions,
    )
    
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/classical_svm.joblib",
    )

    report = classification_report(
        y_val,
        predictions,
        output_dict=True,
    )

    metrics_df = pd.DataFrame(
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
    )

    metrics_df.to_csv(
        "artifacts/classical/metrics.csv",
        index=False,
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        classification_report(
            y_val,
            predictions,
        )
    )


if __name__ == "__main__":
    main()