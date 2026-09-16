#!/usr/bin/env python3
"""Summarize validation-only tuning candidates without reading outer-test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


CANDIDATES = (
    "tuning_a_baseline",
    "tuning_b_regularized",
    "tuning_c_lower_lr",
    "tuning_d_class_weighted",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    args = parser.parse_args()

    baseline_dir = args.runs_root / CANDIDATES[0]
    rows = []
    split_checks = []

    for candidate in CANDIDATES:
        run_dir = args.runs_root / candidate
        config = read_json(run_dir / "run_config.json")
        if not config.get("selection_only"):
            raise RuntimeError(f"Candidate is not validation-only: {candidate}")
        if list(run_dir.rglob("predictions.csv")):
            raise RuntimeError(f"Outer-test predictions exist for candidate: {candidate}")

        predictions = []
        metrics = []
        for fold in range(1, args.fold_count + 1):
            fold_dir = run_dir / "repeat_1" / f"fold_{fold}"
            predictions.append(pd.read_csv(fold_dir / "validation_predictions.csv"))
            metrics.append(read_json(fold_dir / "validation_metrics.json"))

            baseline_split = read_json(
                baseline_dir / "repeat_1" / f"fold_{fold}" / "split.json"
            )
            candidate_split = read_json(fold_dir / "split.json")
            split_checks.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "exact_split_match": candidate_split == baseline_split,
                }
            )

        prediction_frame = pd.concat(predictions, ignore_index=True)
        metric_frame = pd.DataFrame(metrics)
        rows.append(
            {
                "candidate": candidate,
                "learning_rate": config["learning_rate"],
                "dropout": config["dropout"],
                "weight_decay": config["weight_decay"],
                "class_weighted_loss": config["class_weighted_loss"],
                "validation_observation_count": int(len(prediction_frame)),
                "mean_fold_validation_auc": float(metric_frame["validation_auc"].mean()),
                "sd_fold_validation_auc": float(metric_frame["validation_auc"].std(ddof=1)),
                "pooled_validation_auc": float(
                    roc_auc_score(
                        prediction_frame["Label"], prediction_frame["Probability_1"]
                    )
                ),
                "pooled_validation_log_loss": float(
                    log_loss(
                        prediction_frame["Label"],
                        prediction_frame["Probability_1"],
                        labels=[0, 1],
                    )
                ),
                "mean_best_validation_loss": float(metric_frame["best_val_loss"].mean()),
                "mean_epochs": float(metric_frame["epochs_ran"].mean()),
            }
        )

    if not all(row["exact_split_match"] for row in split_checks):
        raise RuntimeError("Candidate splits are not identical")

    metrics_frame = pd.DataFrame(rows)
    best_auc = metrics_frame["mean_fold_validation_auc"].max()
    eligible = metrics_frame[
        metrics_frame["mean_fold_validation_auc"] >= best_auc - 0.01
    ]
    selected = eligible.sort_values(
        ["mean_best_validation_loss", "pooled_validation_log_loss", "candidate"]
    ).iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output_dir / "tuning_validation_metrics.csv", index=False)
    pd.DataFrame(split_checks).to_csv(
        args.output_dir / "tuning_validation_split_checks.csv", index=False
    )
    result = {
        "status": "complete",
        "selection_scope": "validation predictions only; outer-test predictions were not generated",
        "selection_rule": (
            "retain candidates within 0.01 of the highest mean fold validation AUC, "
            "then minimize mean best validation loss"
        ),
        "all_splits_match_exactly": True,
        "outer_test_prediction_file_count": 0,
        "candidate_metrics": rows,
        "selected_candidate": selected["candidate"],
        "selected_configuration": {
            "learning_rate": float(selected["learning_rate"]),
            "dropout": float(selected["dropout"]),
            "weight_decay": float(selected["weight_decay"]),
            "class_weighted_loss": bool(selected["class_weighted_loss"]),
        },
    }
    with (args.output_dir / "tuning_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
