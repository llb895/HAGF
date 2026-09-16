#!/usr/bin/env python3
"""Compare one component ablation with the full model on matched CV splits."""

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


def paired_bootstrap_delta(
    labels: np.ndarray,
    full_probabilities: np.ndarray,
    ablated_probabilities: np.ndarray,
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
        deltas[iteration] = roc_auc_score(
            labels[sampled], ablated_probabilities[sampled]
        ) - roc_auc_score(labels[sampled], full_probabilities[sampled])
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--ablated", type=Path, required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    split_rows = []
    for repeat in range(1, 6):
        for fold in range(1, 11):
            full_split = read_json(
                args.full / f"repeat_{repeat}" / f"fold_{fold}" / "split.json"
            )
            ablated_split = read_json(
                args.ablated / f"repeat_{repeat}" / f"fold_{fold}" / "split.json"
            )
            split_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "exact_split_match": full_split == ablated_split,
                }
            )
    if not all(row["exact_split_match"] for row in split_rows):
        raise RuntimeError("Full and ablated model splits do not match")

    full_summary = read_json(args.full / "summary.json")
    ablated_summary = read_json(args.ablated / "summary.json")
    full_predictions = pd.read_csv(args.full / "sample_averaged_predictions.csv")[
        ["ID", "Label", "Probability_1"]
    ].rename(columns={"Probability_1": "full_probability"})
    ablated_predictions = pd.read_csv(args.ablated / "sample_averaged_predictions.csv")[
        ["ID", "Label", "Probability_1"]
    ].rename(columns={"Probability_1": "ablated_probability"})
    paired = full_predictions.merge(
        ablated_predictions, on=["ID", "Label"], validate="one_to_one"
    )

    labels = paired["Label"].to_numpy()
    full_probability = paired["full_probability"].to_numpy()
    ablated_probability = paired["ablated_probability"].to_numpy()
    lower, upper = paired_bootstrap_delta(
        labels,
        full_probability,
        ablated_probability,
        args.bootstrap_iterations,
        seed=20260910,
    )
    repeat_deltas = np.asarray(ablated_summary["repeat_auc_values"]) - np.asarray(
        full_summary["repeat_auc_values"]
    )

    result = {
        "status": "complete",
        "component_removed": args.component,
        "all_splits_match_exactly": True,
        "matched_split_count": len(split_rows),
        "sample_count": int(len(paired)),
        "full_model": {
            "repeat_auc_mean": full_summary["repeat_auc_mean"],
            "repeat_auc_sd": full_summary["repeat_auc_sample_sd"],
            "sample_averaged_auc": full_summary["sample_averaged_auc"],
            "parameter_count": full_summary["parameter_count"],
            "mean_fold_runtime_seconds": full_summary["mean_fold_runtime_seconds"],
        },
        "ablated_model": {
            "repeat_auc_mean": ablated_summary["repeat_auc_mean"],
            "repeat_auc_sd": ablated_summary["repeat_auc_sample_sd"],
            "sample_averaged_auc": ablated_summary["sample_averaged_auc"],
            "parameter_count": ablated_summary["parameter_count"],
            "mean_fold_runtime_seconds": ablated_summary["mean_fold_runtime_seconds"],
        },
        "repeat_auc_delta_ablated_minus_full_mean": float(repeat_deltas.mean()),
        "repeat_auc_delta_ablated_minus_full_sd": float(repeat_deltas.std(ddof=1)),
        "sample_averaged_auc_delta_ablated_minus_full": float(
            roc_auc_score(labels, ablated_probability)
            - roc_auc_score(labels, full_probability)
        ),
        "sample_averaged_auc_delta_bootstrap_95_ci": [lower, upper],
        "bootstrap_iterations": args.bootstrap_iterations,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "matched_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "split_checks.csv", index=False)
    with (args.output_dir / "component_ablation_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
