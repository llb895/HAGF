"""Plot the historical HRA003209 AUC audit used for Reviewer 1, Comment 5."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CV_COLOR = "#147D73"
INDEPENDENT_COLOR = "#D9772A"
TEXT_COLOR = "#26333D"
GRID_COLOR = "#D5DDE3"


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"ID", "Label", "Probability"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing required columns in {path}")
    result = frame[["ID", "Label", "Probability"]].copy()
    result["Label"] = pd.to_numeric(result["Label"], errors="raise").astype(int)
    result["Probability"] = pd.to_numeric(
        result["Probability"], errors="raise"
    ).astype(float)
    if set(result["Label"].unique()) != {0, 1}:
        raise ValueError(f"binary classes are incomplete in {path}")
    return result


def repeat_from_name(path: Path) -> int:
    match = re.fullmatch(r"(\d+)_(\d+)_fold", path.stem)
    if match is None:
        raise ValueError(f"unexpected prediction filename: {path.name}")
    return int(match.group(1))


def bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        values[index] = roc_auc_score(labels[sampled], probabilities[sampled])
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dir", type=Path, required=True)
    parser.add_argument("--independent-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    records: list[tuple[int, pd.DataFrame]] = []
    for path in sorted(args.cross_dir.glob("*.csv")):
        records.append((repeat_from_name(path), read_prediction(path)))
    if len(records) != 50:
        raise ValueError(f"expected 50 cross-validation files, found {len(records)}")

    repeat_aucs = []
    for repeat in range(1, 6):
        frames = [frame for candidate, frame in records if candidate == repeat]
        if len(frames) != 10:
            raise ValueError(f"repeat {repeat} has {len(frames)} folds")
        combined = pd.concat(frames, ignore_index=True)
        repeat_aucs.append(
            roc_auc_score(combined["Label"], combined["Probability"])
        )

    stacked = pd.concat([frame for _, frame in records], ignore_index=True)
    stacked_auc = roc_auc_score(stacked["Label"], stacked["Probability"])
    independent = read_prediction(args.independent_file)
    independent_auc = roc_auc_score(
        independent["Label"], independent["Probability"]
    )
    independent_ci = bootstrap_auc(
        independent["Label"].to_numpy(),
        independent["Probability"].to_numpy(),
        args.bootstrap_iterations,
        20260909,
    )

    repeat_values = np.asarray(repeat_aucs)
    repeat_mean = repeat_values.mean()
    repeat_sd = repeat_values.std(ddof=1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.edgecolor": "#53606B",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.24, wspace=0.34)
    figure.suptitle(
        "Bie et al. multi-cancer cohort (HRA003209): audit of the reported AUC 0.956",
        fontsize=12.2,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.5,
        0.895,
        "Historical saved predictions; no model retraining",
        ha="center",
        color="#74808A",
        fontsize=8.8,
    )

    ax = axes[0]
    repeats = np.arange(1, 6)
    ax.axhspan(
        repeat_mean - repeat_sd,
        repeat_mean + repeat_sd,
        color=CV_COLOR,
        alpha=0.11,
        linewidth=0,
    )
    ax.axhline(repeat_mean, color=CV_COLOR, linestyle="--", linewidth=1.1)
    ax.plot(
        repeats,
        repeat_values,
        color=CV_COLOR,
        marker="o",
        markersize=5.5,
        linewidth=1.5,
        label=f"Repeated CV: {repeat_mean:.4f} +/- {repeat_sd:.4f}",
    )
    asymmetric_error = np.array(
        [[independent_auc - independent_ci[0]], [independent_ci[1] - independent_auc]]
    )
    ax.errorbar(
        [6],
        [independent_auc],
        yerr=asymmetric_error,
        fmt="s",
        color=INDEPENDENT_COLOR,
        ecolor=INDEPENDENT_COLOR,
        capsize=4,
        markersize=5.5,
        linewidth=1.4,
        label=(
            f"Independent: {independent_auc:.4f}\n"
            f"95% CI {independent_ci[0]:.4f}-{independent_ci[1]:.4f}"
        ),
    )
    ax.set_title("A. Repeated CV and independent validation", loc="left", fontweight="bold")
    ax.set_ylabel("ROC AUC")
    ax.set_xticks([1, 2, 3, 4, 5, 6], ["R1", "R2", "R3", "R4", "R5", "Independent"])
    ax.set_xlim(0.6, 6.4)
    ax.set_ylim(0.93, 0.98)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    rounding_low = 0.9555
    rounding_high = 0.9565
    ax.axvspan(rounding_low, rounding_high, color="#EEF6F4", linewidth=0)
    y_positions = [1, 0]
    exact_values = [stacked_auc, independent_auc]
    labels = ["Pooled CV", "Independent"]
    colors = [CV_COLOR, INDEPENDENT_COLOR]
    for y, value, color in zip(y_positions, exact_values, colors):
        ax.plot(value, y, marker="o", markersize=7, color=color, zorder=3)
        ax.vlines(value, y - 0.14, y + 0.14, color=color, linewidth=1.2)
        ax.annotate(
            f"{value:.7f} -> 0.956",
            xy=(value, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_title("B. Why both values display as 0.956", loc="left", fontweight="bold")
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0.95545, 0.95655)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Unrounded ROC AUC")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax.text(
        0.5,
        0.08,
        "Values in [0.9555, 0.9565) round to 0.956",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.4,
        color="#5D6973",
    )

    figure.text(
        0.08,
        0.075,
        "Panel A: AUC was calculated once for each complete 10-fold repeat; the band is mean +/- sample SD. "
        "The independent-test CI is a class-stratified bootstrap interval (10,000 resamples).",
        fontsize=7.6,
        color="#5D6973",
    )
    figure.text(
        0.08,
        0.035,
        "Panel B: the manuscript values are valid three-decimal summaries of two distinct unrounded estimates; "
        "their equality after rounding is not a duplicated result.",
        fontsize=7.6,
        color="#5D6973",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "HRA003209 original AUC audit",
            "Author": "HAGF revision analysis",
            "Subject": "Reviewer 1 Comment 5",
        },
    )
    plt.close(figure)

    print(
        "AUDIT_FIGURE_COMPLETE",
        f"repeat_mean={repeat_mean:.10f}",
        f"repeat_sd={repeat_sd:.10f}",
        f"stacked_auc={stacked_auc:.10f}",
        f"independent_auc={independent_auc:.10f}",
        f"independent_ci={independent_ci}",
        args.output,
    )


if __name__ == "__main__":
    main()
