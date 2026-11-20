"""Plot the recorded CIFAR-10 validation checkpoints across three seeds."""

import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    results = Path(__file__).parent / "results"
    with (results / "validation-by-step.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axis = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for variant, label, color, style in (
        ("baseline", "Dense", "#175d95", "-"),
        ("cst-one", "One CST projection", "#ad470c", "--"),
        ("cst-two", "Two CST projections", "#147a64", "-."),
        ("cst-shortcut", "Two CST + shortcut", "#765895", ":"),
    ):
        trial = [row for row in rows if row["variant"] == variant]
        steps = sorted({int(row["step"]) for row in trial})
        series = [[float(row["top1_pct"]) for row in trial if int(row["step"]) == step] for step in steps]
        axis.plot(steps, [mean(x) for x in series], style, color=color,
                  label=label, marker="o", markersize=3)
        axis.fill_between(steps, [min(x) for x in series], [max(x) for x in series],
                          color=color, alpha=0.10, linewidth=0)
    axis.set(title="CIFAR-10 validation across three seeds", xlabel="Training updates",
             ylabel="Top-1 accuracy (%)")
    axis.grid(axis="y", color="#d9e1e7", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)
    figure.savefig(results / "learning-curves.png", dpi=190, facecolor="white")


if __name__ == "__main__":
    main()
