import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "Cat",
    "Dog",
    "Opossum",
    "Raccoon"
]

df = pd.read_csv(
    "artifacts/deep/confusion_matrix.csv",
    header=None
)

plt.figure(figsize=(7,6))

plt.imshow(
    df,
    interpolation="nearest"
)

plt.title(
    "MobileNetV3 Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

for i in range(df.shape[0]):
    for j in range(df.shape[1]):

        plt.text(
            j,
            i,
            str(df.iloc[i,j]),
            ha="center",
            va="center",
            color="black"
        )

plt.tight_layout()

plt.savefig(
    "artifacts/deep/confusion_matrix_labeled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()