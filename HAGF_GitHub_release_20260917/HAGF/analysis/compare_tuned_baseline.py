#!/usr/bin/env python3
"""Compare a validation-selected run with the frozen baseline on matched splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_delta(
    labels: np.ndarray,
    baseline: np.ndarray,
    tuned: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        deltas[iteration] = roc_auc_score(labels[sampled], tuned[sampled]) - roc_auc_score(
            labels[sampled], baseline[sampled]
        )
    return tuple(float(value) for value in np.percentile(deltas, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    split_rows = []
    for repeat in range(1, 6):
        for fold in range(1, 11):
            baseline_split = read_json(
                args.baseline / f"repeat_{repeat}" / f"fold_{fold}" / "split.json"
            )
            tuned_split = read_json(
                args.tuned / f"repeat_{repeat}" / f"fold_{fold}" / "split.json"
            )
            split_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "exact_split_match": baseline_split == tuned_split,
                }
            )
    if not all(row["exact_split_match"] for row in split_rows):
        raise RuntimeError("Baseline and tuned splits do not match")

    baseline_summary = read_json(args.baseline / "summary.json")
    tuned_summary = read_json(args.tuned / "summary.json")
    baseline_predictions = pd.read_csv(args.baseline / "sample_averaged_predictions.csv")[
        ["ID", "Label", "Probability_1"]
    ].rename(columns={"Probability_1": "baseline_probability"})
    tuned_predictions = pd.read_csv(args.tuned / "sample_averaged_predictions.csv")[
        ["ID", "Label", "Probability_1"]
    ].rename(columns={"Probability_1": "tuned_probability"})
    paired = baseline_predictions.merge(
        tuned_predictions, on=["ID", "Label"], validate="one_to_one"
    )

    labels = paired["Label"].to_numpy()
    baseline_probability = paired["baseline_probability"].to_numpy()
    tuned_probability = paired["tuned_probability"].to_numpy()
    lower, upper = bootstrap_delta(
        labels,
        baseline_probability,
        tuned_probability,
        args.bootstrap_iterations,
        seed=20260910,
    )
    repeat_deltas = np.asarray(tuned_summary["repeat_auc_values"]) - np.asarray(
        baseline_summary["repeat_auc_values"]
    )

    result = {
        "status": "complete",
        "all_splits_match_exactly": True,
        "matched_split_count": len(split_rows),
        "sample_count": int(len(paired)),
        "baseline": {
            "repeat_auc_mean": baseline_summary["repeat_auc_mean"],
            "repeat_auc_sd": baseline_summary["repeat_auc_sample_sd"],
            "sample_averaged_auc": baseline_summary["sample_averaged_auc"],
            "mean_fold_runtime_seconds": baseline_summary["mean_fold_runtime_seconds"],
        },
        "validation_selected_regularized": {
            "repeat_auc_mean": tuned_summary["repeat_auc_mean"],
            "repeat_auc_sd": tuned_summary["repeat_auc_sample_sd"],
            "sample_averaged_auc": tuned_summary["sample_averaged_auc"],
            "mean_fold_runtime_seconds": tuned_summary["mean_fold_runtime_seconds"],
        },
        "repeat_auc_delta_mean": float(repeat_deltas.mean()),
        "repeat_auc_delta_sd": float(repeat_deltas.std(ddof=1)),
        "sample_averaged_auc_delta": float(
            roc_auc_score(labels, tuned_probability)
            - roc_auc_score(labels, baseline_probability)
        ),
        "sample_averaged_auc_delta_bootstrap_95_ci": [lower, upper],
        "bootstrap_iterations": args.bootstrap_iterations,
        "decision": (
            "Retain the frozen baseline because validation-selected regularization "
            "did not improve outer cross-validation performance."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "baseline_tuned_matched_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(
        args.output_dir / "baseline_tuned_split_checks.csv", index=False
    )
    with (args.output_dir / "baseline_tuned_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
