#!/usr/bin/env python3
"""Summarize repeated cross-validation predictions from a HAGF run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def numbered_dirs(root: Path, prefix: str) -> list[Path]:
    return sorted(
        (path for path in root.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )


def stratified_bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    aucs = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        aucs[iteration] = roc_auc_score(labels[sampled], probabilities[sampled])

    lower, upper = np.percentile(aucs, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    args = parser.parse_args()

    repeat_rows: list[dict[str, float | int]] = []
    all_predictions: list[pd.DataFrame] = []

    repeat_dirs = numbered_dirs(args.run_dir, "repeat")
    if not repeat_dirs:
        raise RuntimeError(f"No repeat directories found in {args.run_dir}")

    for repeat_dir in repeat_dirs:
        repeat_number = int(repeat_dir.name.rsplit("_", 1)[1])
        prediction_frames = []
        for fold_dir in numbered_dirs(repeat_dir, "fold"):
            prediction_path = fold_dir / "predictions.csv"
            if not prediction_path.exists():
                raise RuntimeError(f"Missing prediction file: {prediction_path}")
            frame = pd.read_csv(prediction_path)
            frame["Repeat"] = repeat_number
            frame["Fold"] = int(fold_dir.name.rsplit("_", 1)[1])
            prediction_frames.append(frame)

        repeat_predictions = pd.concat(prediction_frames, ignore_index=True)
        if repeat_predictions["ID"].duplicated().any():
            raise RuntimeError(f"Duplicate sample IDs within repeat {repeat_number}")
        repeat_auc = roc_auc_score(
            repeat_predictions["Label"], repeat_predictions["Probability_1"]
        )
        repeat_rows.append(
            {
                "repeat": repeat_number,
                "sample_count": int(len(repeat_predictions)),
                "auc": float(repeat_auc),
            }
        )
        all_predictions.append(repeat_predictions)

    predictions = pd.concat(all_predictions, ignore_index=True)
    label_counts = predictions.groupby("ID")["Label"].nunique()
    if (label_counts != 1).any():
        raise RuntimeError("At least one sample has inconsistent labels across repeats")

    averaged = (
        predictions.groupby("ID", as_index=False)
        .agg(Label=("Label", "first"), Probability_1=("Probability_1", "mean"))
        .sort_values("ID")
    )
    averaged["Probability_0"] = 1.0 - averaged["Probability_1"]
    averaged = averaged[["ID", "Label", "Probability_0", "Probability_1"]]

    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_auc_mean = float(repeat_metrics["auc"].mean())
    repeat_auc_sd = float(repeat_metrics["auc"].std(ddof=1))
    stacked_auc = float(roc_auc_score(predictions["Label"], predictions["Probability_1"]))
    averaged_auc = float(roc_auc_score(averaged["Label"], averaged["Probability_1"]))
    ci_lower, ci_upper = stratified_bootstrap_auc(
        averaged["Label"].to_numpy(),
        averaged["Probability_1"].to_numpy(),
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )

    fold_metrics_path = args.run_dir / "fold_metrics.csv"
    fold_metrics = pd.read_csv(fold_metrics_path)
    expected_folds = sum(len(numbered_dirs(path, "fold")) for path in repeat_dirs)
    if len(fold_metrics) != expected_folds:
        raise RuntimeError(
            f"fold_metrics.csv has {len(fold_metrics)} rows; expected {expected_folds}"
        )

    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "status": "complete",
        "repeat_count": int(len(repeat_dirs)),
        "fold_count": int(expected_folds),
        "unique_sample_count": int(averaged["ID"].nunique()),
        "prediction_row_count": int(len(predictions)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in averaged["Label"].value_counts().sort_index().items()
        },
        "repeat_auc_values": [float(value) for value in repeat_metrics["auc"]],
        "repeat_auc_mean": repeat_auc_mean,
        "repeat_auc_sample_sd": repeat_auc_sd,
        "stacked_prediction_auc": stacked_auc,
        "sample_averaged_auc": averaged_auc,
        "sample_averaged_auc_bootstrap_95_ci": [ci_lower, ci_upper],
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "parameter_count": int(fold_metrics["parameter_count"].iloc[0]),
        "fold_auc_mean_not_primary": float(fold_metrics["auc"].mean()),
        "fold_auc_sample_sd_not_primary": float(fold_metrics["auc"].std(ddof=1)),
        "epochs_mean": float(fold_metrics["epochs_ran"].mean()),
        "epochs_median": float(fold_metrics["epochs_ran"].median()),
        "epochs_range": [
            int(fold_metrics["epochs_ran"].min()),
            int(fold_metrics["epochs_ran"].max()),
        ],
        "total_fold_runtime_seconds": float(fold_metrics["runtime_seconds"].sum()),
        "mean_fold_runtime_seconds": float(fold_metrics["runtime_seconds"].mean()),
    }

    repeat_metrics.to_csv(args.run_dir / "repeat_metrics.csv", index=False)
    averaged.to_csv(args.run_dir / "sample_averaged_predictions.csv", index=False)
    with (args.run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
