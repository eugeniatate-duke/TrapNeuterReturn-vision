import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "artifacts/deep/confusion_matrix.csv",
    index_col=0
)

CLASS_NAMES = [
    label.title()
    for label in df.columns
]

plt.figure(figsize=(7,6))

plt.imshow(
    df.values,
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
            str(df.values[i,j]),
            ha="center",
            va="center",
            color="black"
        )

plt.tight_layout()

print(df)

print(df.shape)

plt.savefig(
    "artifacts/deep/confusion_matrix_labeled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()