"""Audit historical APBC binary-prediction files without retraining models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


COHORTS = ("CRA001537", "PRJNA929650", "HRA003209")
MAIN_CV_DIRECTORIES = {
    "CRA001537": "CRA001537_5_10folds",
    "PRJNA929650": "PRJNA929650_5_10folds_cross",
    "HRA003209": "HRA003209_5_10folds_cross",
}
MAIN_VALIDATION_DIRECTORIES = {
    "PRJNA929650": "PRJNA929650_5_10folds_val",
    "HRA003209": "HRA003209_5_10folds_val",
}
LABEL_COLUMNS = ("label", "true label", "true_label", "y_true")
PROBABILITY_COLUMNS = (
    "probability",
    "probability_1",
    "predicted probability",
    "predicted_probability",
    "y_score",
)


def resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return str(normalized[candidate])
    return None


def read_binary_prediction(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    label_column = resolve_column(frame, LABEL_COLUMNS)
    probability_column = resolve_column(frame, PROBABILITY_COLUMNS)
    if label_column is None or probability_column is None:
        return None
    result = pd.DataFrame(
        {
            "label": pd.to_numeric(frame[label_column], errors="coerce"),
            "probability": pd.to_numeric(frame[probability_column], errors="coerce"),
        }
    ).dropna()
    id_column = resolve_column(frame, ("id", "sample", "sample_id"))
    if id_column is not None:
        result["id"] = frame.loc[result.index, id_column].astype(str).to_numpy()
    labels = set(result["label"].astype(int).unique().tolist())
    if len(result) < 2 or labels != {0, 1}:
        return None
    return result.reset_index(drop=True)


def bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    seed: int = 20260909,
    iterations: int = 10_000,
) -> tuple[float, float] | None:
    if iterations <= 0:
        return None
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        values[index] = roc_auc_score(labels[sampled], probabilities[sampled])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def fold_coordinates(path: Path) -> tuple[int, int] | None:
    match = re.search(r"(?:^|_)(\d+)_(\d+)(?:_fold)?$", path.stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def audit_cv_directory(path: Path, bootstrap_iterations: int) -> dict | None:
    records = []
    for csv_path in sorted(path.glob("*.csv")):
        prediction = read_binary_prediction(csv_path)
        coordinates = fold_coordinates(csv_path)
        if prediction is None or coordinates is None:
            continue
        repeat, fold = coordinates
        records.append((repeat, fold, csv_path, prediction))
    if len(records) < 10:
        return None

    fold_aucs = []
    repeat_aucs = []
    repeat_counts = {}
    for repeat, fold, _, prediction in records:
        fold_aucs.append(
            {
                "repeat": repeat,
                "fold": fold,
                "auc": float(
                    roc_auc_score(prediction["label"], prediction["probability"])
                ),
                "n": len(prediction),
            }
        )
    for repeat in sorted({item[0] for item in records}):
        repeat_frames = [item[3] for item in records if item[0] == repeat]
        combined = pd.concat(repeat_frames, ignore_index=True)
        repeat_counts[str(repeat)] = len(repeat_frames)
        repeat_aucs.append(
            float(roc_auc_score(combined["label"], combined["probability"]))
        )

    combined = pd.concat([item[3] for item in records], ignore_index=True)
    summary = {
        "directory": str(path),
        "file_count": len(records),
        "repeat_fold_counts": repeat_counts,
        "row_count": len(combined),
        "fold_auc_mean": float(np.mean([item["auc"] for item in fold_aucs])),
        "fold_auc_sd": float(np.std([item["auc"] for item in fold_aucs], ddof=1)),
        "repeat_auc_values": repeat_aucs,
        "repeat_auc_mean": float(np.mean(repeat_aucs)),
        "repeat_auc_sd": float(np.std(repeat_aucs, ddof=1)) if len(repeat_aucs) > 1 else None,
        "stacked_auc": float(roc_auc_score(combined["label"], combined["probability"])),
        "fold_aucs": fold_aucs,
    }
    if "id" in combined.columns:
        by_id = (
            combined.groupby("id", as_index=False)
            .agg(label=("label", "first"), probability=("probability", "mean"), n=("id", "size"))
        )
        if by_id.groupby("id")["label"].nunique().max() == 1:
            labels = by_id["label"].to_numpy(dtype=int)
            probabilities = by_id["probability"].to_numpy(dtype=float)
            summary["unique_sample_count"] = len(by_id)
            summary["predictions_per_sample"] = sorted(by_id["n"].unique().tolist())
            summary["sample_aggregated_auc"] = float(roc_auc_score(labels, probabilities))
            summary["sample_aggregated_auc_bootstrap_95ci"] = bootstrap_auc(
                labels, probabilities, iterations=bootstrap_iterations
            )
    return summary


def audit_independent_file(path: Path, bootstrap_iterations: int) -> dict | None:
    prediction = read_binary_prediction(path)
    if prediction is None:
        return None
    labels = prediction["label"].to_numpy(dtype=int)
    probabilities = prediction["probability"].to_numpy(dtype=float)
    return {
        "file": str(path),
        "sample_count": len(prediction),
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "auc": float(roc_auc_score(labels, probabilities)),
        "bootstrap_95ci": bootstrap_auc(
            labels, probabilities, iterations=bootstrap_iterations
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--main-results-only", action="store_true")
    args = parser.parse_args()

    report = {
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": 20260909,
        "cohorts": {},
    }
    for cohort in COHORTS:
        plot_root = args.subject_root / cohort / "plot"
        cv_results = []
        for directory in sorted(path for path in plot_root.iterdir() if path.is_dir()):
            if (
                args.main_results_only
                and directory.name != MAIN_CV_DIRECTORIES[cohort]
            ):
                continue
            result = audit_cv_directory(directory, args.bootstrap_iterations)
            if result is not None:
                cv_results.append(result)
        independent_results = []
        for path in sorted(plot_root.rglob("model_val.csv")):
            if args.main_results_only and path.parent.name != MAIN_VALIDATION_DIRECTORIES.get(
                cohort
            ):
                continue
            result = audit_independent_file(path, args.bootstrap_iterations)
            if result is not None:
                independent_results.append(result)
        report["cohorts"][cohort] = {
            "cross_validation_candidates": cv_results,
            "independent_validation_candidates": independent_results,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"AUDIT_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
