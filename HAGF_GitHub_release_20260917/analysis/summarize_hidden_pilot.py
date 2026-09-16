#!/usr/bin/env python3
"""Compare hidden-size pilot runs on identical cross-validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_variant(run_dir: Path, fold_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    metrics = []
    for fold in range(1, fold_count + 1):
        fold_dir = run_dir / "repeat_1" / f"fold_{fold}"
        prediction = pd.read_csv(fold_dir / "predictions.csv")
        prediction["fold"] = fold
        predictions.append(prediction)

        with (fold_dir / "metrics.json").open(encoding="utf-8") as handle:
            metric = json.load(handle)
        metric["fold"] = fold
        metrics.append(metric)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def load_split(run_dir: Path, fold: int) -> dict[str, list[str]]:
    path = run_dir / "repeat_1" / f"fold_{fold}" / "split.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def paired_bootstrap_delta(
    merged: pd.DataFrame,
    probability_column: str,
    baseline_column: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    labels = merged["Label"].to_numpy()
    candidate = merged[probability_column].to_numpy()
    baseline = merged[baseline_column].to_numpy()
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    deltas = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        deltas[iteration] = roc_auc_score(labels[sampled], candidate[sampled]) - roc_auc_score(
            labels[sampled], baseline[sampled]
        )
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hidden64", type=Path, required=True)
    parser.add_argument("--hidden200", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    run_dirs = {100: args.baseline, 64: args.hidden64, 200: args.hidden200}
    split_checks = []
    for hidden_size in (64, 200):
        for fold in range(1, args.fold_count + 1):
            matches = load_split(run_dirs[hidden_size], fold) == load_split(args.baseline, fold)
            split_checks.append(
                {"hidden_size": hidden_size, "fold": fold, "exact_split_match": matches}
            )
    if not all(row["exact_split_match"] for row in split_checks):
        raise RuntimeError("At least one pilot split differs from the hidden-size 100 baseline")

    variant_rows = []
    prediction_tables: dict[int, pd.DataFrame] = {}
    for hidden_size in (64, 100, 200):
        predictions, metrics = load_variant(run_dirs[hidden_size], args.fold_count)
        prediction_tables[hidden_size] = predictions
        variant_rows.append(
            {
                "hidden_size": hidden_size,
                "sample_count": int(len(predictions)),
                "positive_count": int(predictions["Label"].sum()),
                "pooled_auc": float(
                    roc_auc_score(predictions["Label"], predictions["Probability_1"])
                ),
                "fold_auc_mean_not_primary": float(metrics["auc"].mean()),
                "fold_auc_sd_not_primary": float(metrics["auc"].std(ddof=1)),
                "parameter_count": int(metrics["parameter_count"].iloc[0]),
                "mean_epochs": float(metrics["epochs_ran"].mean()),
                "total_runtime_seconds": float(metrics["runtime_seconds"].sum()),
            }
        )

    merged = prediction_tables[100][["ID", "Label", "Probability_1"]].rename(
        columns={"Probability_1": "probability_h100"}
    )
    for hidden_size in (64, 200):
        candidate = prediction_tables[hidden_size][["ID", "Label", "Probability_1"]].rename(
            columns={"Probability_1": f"probability_h{hidden_size}"}
        )
        merged = merged.merge(candidate, on=["ID", "Label"], validate="one_to_one")

    baseline_auc = next(row["pooled_auc"] for row in variant_rows if row["hidden_size"] == 100)
    delta_rows = []
    for hidden_size in (64, 200):
        candidate_auc = next(
            row["pooled_auc"] for row in variant_rows if row["hidden_size"] == hidden_size
        )
        ci_lower, ci_upper = paired_bootstrap_delta(
            merged,
            f"probability_h{hidden_size}",
            "probability_h100",
            args.bootstrap_iterations,
            seed=20260909 + hidden_size,
        )
        delta_rows.append(
            {
                "hidden_size": hidden_size,
                "auc_delta_vs_h100": float(candidate_auc - baseline_auc),
                "paired_bootstrap_delta_95_ci": [ci_lower, ci_upper],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = pd.DataFrame(variant_rows).sort_values("hidden_size")
    variants.to_csv(args.output_dir / "hidden_size_pilot_metrics.csv", index=False)
    pd.DataFrame(split_checks).to_csv(args.output_dir / "hidden_size_split_checks.csv", index=False)
    merged.to_csv(args.output_dir / "hidden_size_matched_predictions.csv", index=False)

    summary = {
        "status": "complete",
        "design": "first five folds of repeat 1 from the same 10-fold split",
        "pilot_only": True,
        "all_splits_match_exactly": True,
        "fold_count": args.fold_count,
        "variant_metrics": variant_rows,
        "deltas_vs_hidden100": delta_rows,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    with (args.output_dir / "hidden_size_pilot_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
