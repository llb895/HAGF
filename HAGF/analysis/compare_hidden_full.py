#!/usr/bin/env python3
"""Compare full hidden-size runs using matched splits and paired bootstrap."""

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
    frame: pd.DataFrame,
    candidate_column: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    labels = frame["Label"].to_numpy()
    candidate = frame[candidate_column].to_numpy()
    baseline = frame["probability_h100"].to_numpy()
    class_indices = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        deltas[iteration] = roc_auc_score(labels[sampled], candidate[sampled]) - roc_auc_score(
            labels[sampled], baseline[sampled]
        )
    return tuple(float(value) for value in np.percentile(deltas, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden64", type=Path, required=True)
    parser.add_argument("--hidden100", type=Path, required=True)
    parser.add_argument("--hidden200", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    runs = {64: args.hidden64, 100: args.hidden100, 200: args.hidden200}
    summaries = {hidden: read_json(path / "summary.json") for hidden, path in runs.items()}

    split_rows = []
    for hidden in (64, 200):
        for repeat in range(1, 6):
            for fold in range(1, 11):
                candidate = read_json(runs[hidden] / f"repeat_{repeat}" / f"fold_{fold}" / "split.json")
                baseline = read_json(runs[100] / f"repeat_{repeat}" / f"fold_{fold}" / "split.json")
                split_rows.append(
                    {
                        "hidden_size": hidden,
                        "repeat": repeat,
                        "fold": fold,
                        "exact_split_match": candidate == baseline,
                    }
                )
    if not all(row["exact_split_match"] for row in split_rows):
        raise RuntimeError("At least one full-run split does not match hidden size 100")

    merged = None
    rows = []
    for hidden in (64, 100, 200):
        predictions = pd.read_csv(runs[hidden] / "sample_averaged_predictions.csv")[
            ["ID", "Label", "Probability_1"]
        ].rename(columns={"Probability_1": f"probability_h{hidden}"})
        merged = predictions if merged is None else merged.merge(
            predictions, on=["ID", "Label"], validate="one_to_one"
        )
        summary = summaries[hidden]
        rows.append(
            {
                "hidden_size": hidden,
                "parameter_count": summary["parameter_count"],
                "repeat_auc_mean": summary["repeat_auc_mean"],
                "repeat_auc_sd": summary["repeat_auc_sample_sd"],
                "sample_averaged_auc": summary["sample_averaged_auc"],
                "sample_averaged_ci_lower": summary[
                    "sample_averaged_auc_bootstrap_95_ci"
                ][0],
                "sample_averaged_ci_upper": summary[
                    "sample_averaged_auc_bootstrap_95_ci"
                ][1],
                "mean_fold_runtime_seconds": summary["mean_fold_runtime_seconds"],
            }
        )

    assert merged is not None
    baseline_auc = roc_auc_score(merged["Label"], merged["probability_h100"])
    delta_rows = []
    for hidden in (64, 200):
        candidate_auc = roc_auc_score(merged["Label"], merged[f"probability_h{hidden}"])
        lower, upper = paired_bootstrap_delta(
            merged,
            f"probability_h{hidden}",
            args.bootstrap_iterations,
            seed=20260910 + hidden,
        )
        repeat_deltas = np.asarray(summaries[hidden]["repeat_auc_values"]) - np.asarray(
            summaries[100]["repeat_auc_values"]
        )
        delta_rows.append(
            {
                "hidden_size": hidden,
                "sample_averaged_auc_delta_vs_h100": float(candidate_auc - baseline_auc),
                "paired_bootstrap_delta_95_ci": [lower, upper],
                "repeat_auc_delta_mean_vs_h100": float(repeat_deltas.mean()),
                "repeat_auc_delta_sd_vs_h100": float(repeat_deltas.std(ddof=1)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "hidden_size_full_metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.output_dir / "hidden_size_full_deltas.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "hidden_size_full_split_checks.csv", index=False)
    merged.to_csv(args.output_dir / "hidden_size_full_matched_predictions.csv", index=False)

    result = {
        "status": "complete",
        "design": "five repeated stratified 10-fold cross-validations with identical splits",
        "all_100_split_comparisons_match_exactly": True,
        "split_comparisons": len(split_rows),
        "variant_metrics": rows,
        "deltas_vs_hidden100": delta_rows,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    with (args.output_dir / "hidden_size_full_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
