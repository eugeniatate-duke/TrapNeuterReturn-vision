import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "artifacts/experiments/robustness_results.csv"
)

plt.figure(figsize=(6,4))
plt.bar(
    df["condition"],
    df["accuracy"]
)

plt.ylabel("Accuracy")
plt.title(
    "Robustness Experiment"
)

plt.tight_layout()

plt.savefig(
    "artifacts/experiments/robustness_plot.png"
)