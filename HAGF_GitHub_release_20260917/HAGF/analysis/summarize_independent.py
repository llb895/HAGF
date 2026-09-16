#!/usr/bin/env python3
"""Summarize repeated-fold predictions on a fixed independent cohort."""

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


def average_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    predictions = pd.concat(frames, ignore_index=True)
    label_counts = predictions.groupby("ID")["Label"].nunique()
    if (label_counts != 1).any():
        raise RuntimeError("Independent labels are inconsistent across models")
    averaged = (
        predictions.groupby("ID", as_index=False)
        .agg(
            Label=("Label", "first"),
            Probability_1=("Probability_1", "mean"),
            Model_count=("ID", "size"),
        )
        .sort_values("ID")
    )
    averaged["Probability_0"] = 1.0 - averaged["Probability_1"]
    return averaged[
        ["ID", "Label", "Probability_0", "Probability_1", "Model_count"]
    ]


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
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    args = parser.parse_args()

    repeat_dirs = numbered_dirs(args.run_dir, "repeat")
    if not repeat_dirs:
        raise RuntimeError(f"No repeat directories found in {args.run_dir}")

    repeat_rows: list[dict[str, float | int]] = []
    all_frames: list[pd.DataFrame] = []
    expected_fold_count: int | None = None
    for repeat_dir in repeat_dirs:
        fold_dirs = numbered_dirs(repeat_dir, "fold")
        if expected_fold_count is None:
            expected_fold_count = len(fold_dirs)
        elif len(fold_dirs) != expected_fold_count:
            raise RuntimeError("Repeat directories contain different fold counts")

        frames = []
        for fold_dir in fold_dirs:
            path = fold_dir / "independent_predictions.csv"
            if not path.is_file():
                raise RuntimeError(f"Missing independent prediction file: {path}")
            frame = pd.read_csv(path)
            frames.append(frame)
            all_frames.append(frame)

        averaged = average_predictions(frames)
        model_counts = averaged["Model_count"].unique()
        if len(model_counts) != 1 or model_counts[0] != len(fold_dirs):
            raise RuntimeError("Each independent sample must have one prediction per fold")
        repeat_rows.append(
            {
                "repeat": int(repeat_dir.name.rsplit("_", 1)[1]),
                "sample_count": int(len(averaged)),
                "model_count_per_sample": int(model_counts[0]),
                "ensemble_auc": float(
                    roc_auc_score(averaged["Label"], averaged["Probability_1"])
                ),
            }
        )

    combined = average_predictions(all_frames)
    expected_models = len(repeat_dirs) * int(expected_fold_count or 0)
    model_counts = combined["Model_count"].unique()
    if len(model_counts) != 1 or model_counts[0] != expected_models:
        raise RuntimeError("Each independent sample must have one prediction per model")

    repeat_metrics = pd.DataFrame(repeat_rows)
    combined_auc = float(
        roc_auc_score(combined["Label"], combined["Probability_1"])
    )
    ci_lower, ci_upper = stratified_bootstrap_auc(
        combined["Label"].to_numpy(),
        combined["Probability_1"].to_numpy(),
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    fold_metrics = pd.read_csv(args.run_dir / "fold_metrics.csv")
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "status": "complete",
        "repeat_count": int(len(repeat_dirs)),
        "folds_per_repeat": int(expected_fold_count or 0),
        "model_count": expected_models,
        "sample_count": int(len(combined)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in combined["Label"].value_counts().sort_index().items()
        },
        "repeat_ensemble_auc_values": [
            float(value) for value in repeat_metrics["ensemble_auc"]
        ],
        "repeat_ensemble_auc_mean": float(repeat_metrics["ensemble_auc"].mean()),
        "repeat_ensemble_auc_sample_sd": float(
            repeat_metrics["ensemble_auc"].std(ddof=1)
        ),
        "all_model_ensemble_auc": combined_auc,
        "all_model_ensemble_auc_bootstrap_95_ci": [ci_lower, ci_upper],
        "mean_single_fold_auc_not_primary": float(fold_metrics["independent_auc"].mean()),
        "single_fold_auc_sample_sd_not_primary": float(
            fold_metrics["independent_auc"].std(ddof=1)
        ),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
    }

    repeat_metrics.to_csv(
        args.run_dir / "independent_repeat_metrics.csv", index=False
    )
    combined.to_csv(
        args.run_dir / "independent_sample_averaged_predictions.csv", index=False
    )
    with (args.run_dir / "independent_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
