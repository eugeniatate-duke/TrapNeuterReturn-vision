import pandas as pd
import matplotlib.pyplot as plt

def create_plot(
    csv_path,
    output_path,
    title
):

    df = pd.read_csv(
        csv_path,
        index_col=0
    )

    plt.figure(figsize=(6,5))

    plt.imshow(df)

    plt.title(title)

    plt.xticks(
        range(len(df.columns)),
        df.columns,
        rotation=45
    )

    plt.yticks(
        range(len(df.index)),
        df.index
    )

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            plt.text(
                j,
                i,
                str(df.iloc[i,j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


create_plot(
    "artifacts/classical/confusion_matrix.csv",
    "artifacts/classical/confusion_matrix.png",
    "HOG + SVM Confusion Matrix"
)

create_plot(
    "artifacts/deep/confusion_matrix.csv",
    "artifacts/deep/confusion_matrix.png",
    "MobileNetV3 Confusion Matrix"
)