"""Audit existing tissue-of-origin top-1 and top-2 prediction files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "Healthy",
    1: "BRCA",
    2: "COREAD",
    3: "ESCA",
    4: "STAD",
    5: "LIHC",
    6: "NSCLC",
    7: "PACA",
}


def read_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"ID", "Label", "Pre_Label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns in {path}: {required - set(frame.columns)}")
    output = frame[["ID", "Label", "Pre_Label"]].copy()
    output["Label"] = pd.to_numeric(output["Label"], errors="raise").astype(int)
    output["Pre_Label"] = pd.to_numeric(output["Pre_Label"], errors="raise").astype(int)
    output["Correct"] = output["Label"].eq(output["Pre_Label"])
    return output


def bootstrap_accuracy(
    correct: np.ndarray, seed: int = 20260909, iterations: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.choice(correct, size=len(correct), replace=True)
        values[index] = np.mean(sample)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def stratified_bootstrap_macro_accuracy(
    frame: pd.DataFrame, seed: int = 20260909, iterations: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [
        group["Correct"].to_numpy(dtype=float)
        for _, group in frame.groupby("Label")
    ]
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        values[index] = np.mean(
            [
                np.mean(rng.choice(group, size=len(group), replace=True))
                for group in groups
            ]
        )
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def coordinates(path: Path) -> tuple[int, int]:
    match = re.match(r"(\d+)_(\d+)_fold$", path.stem)
    if match is None:
        raise ValueError(f"unexpected fold filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def audit_cross(path: Path) -> dict:
    records = []
    frames_by_repeat: dict[int, list[pd.DataFrame]] = {}
    for csv_path in sorted(path.glob("*_fold.csv")):
        repeat, fold = coordinates(csv_path)
        frame = read_prediction(csv_path)
        records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "n": len(frame),
                "accuracy": float(frame["Correct"].mean()),
            }
        )
        frames_by_repeat.setdefault(repeat, []).append(frame)
    if len(records) != 50:
        raise ValueError(f"expected 50 fold files in {path}, found {len(records)}")

    repeat_accuracies = []
    repeat_macro_accuracies = []
    pooled_frames = []
    for repeat in sorted(frames_by_repeat):
        combined = pd.concat(frames_by_repeat[repeat], ignore_index=True)
        pooled_frames.append(combined)
        repeat_accuracies.append(float(combined["Correct"].mean()))
        repeat_macro_accuracies.append(
            float(combined.groupby("Label")["Correct"].mean().mean())
        )
    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled_class_metrics = [
        {
            "label": label,
            "class": CLASS_NAMES[label],
            "n_predictions": int(len(group)),
            "accuracy": float(group["Correct"].mean()),
        }
        for label, group in pooled.groupby("Label")
    ]
    return {
        "directory": str(path),
        "fold_file_count": len(records),
        "repeat_accuracy_values": repeat_accuracies,
        "repeat_accuracy_mean": float(np.mean(repeat_accuracies)),
        "repeat_accuracy_sd": float(np.std(repeat_accuracies, ddof=1)),
        "repeat_macro_accuracy_values": repeat_macro_accuracies,
        "repeat_macro_accuracy_mean": float(np.mean(repeat_macro_accuracies)),
        "repeat_macro_accuracy_sd": float(np.std(repeat_macro_accuracies, ddof=1)),
        "fold_accuracy_mean": float(np.mean([item["accuracy"] for item in records])),
        "fold_accuracy_sd": float(np.std([item["accuracy"] for item in records], ddof=1)),
        "pooled_class_metrics": pooled_class_metrics,
        "fold_metrics": records,
    }


def audit_validation(path: Path) -> dict:
    frame = read_prediction(path)
    class_metrics = []
    for label, name in CLASS_NAMES.items():
        subset = frame.loc[frame["Label"] == label]
        class_metrics.append(
            {
                "label": label,
                "class": name,
                "n": len(subset),
                "accuracy": float(subset["Correct"].mean()),
            }
        )
    correct = frame["Correct"].to_numpy(dtype=float)
    return {
        "file": str(path),
        "sample_count": len(frame),
        "accuracy": float(np.mean(correct)),
        "macro_accuracy": float(frame.groupby("Label")["Correct"].mean().mean()),
        "macro_accuracy_stratified_bootstrap_95ci": stratified_bootstrap_macro_accuracy(
            frame
        ),
        "bootstrap_95ci": bootstrap_accuracy(correct),
        "class_metrics": class_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plot = args.subject_root / "HRA003209" / "plot"
    report = {
        "cross_validation": {
            "top1": audit_cross(plot / "HAGF_multi_classification"),
            "top2": audit_cross(plot / "HAGF_multi_classification_1and2"),
        },
        "independent_validation": {
            "top1": audit_validation(plot / "HAGF_multi_classification_val" / "model_val.csv"),
            "top2": audit_validation(
                plot / "HAGF_multi_classification_val_1and2" / "model_val.csv"
            ),
        },
        "limitation": (
            "Historical files retain the final top-1 and top-2 correctness outcomes, "
            "not the complete eight-class probability vector."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    for split in ("cross_validation", "independent_validation"):
        for metric in ("top1", "top2"):
            item = report[split][metric]
            print(
                split,
                metric,
                "micro",
                item.get("repeat_accuracy_mean", item.get("accuracy")),
                item.get("repeat_accuracy_sd", item.get("bootstrap_95ci")),
                "macro",
                item.get("repeat_macro_accuracy_mean", item.get("macro_accuracy")),
                item.get("repeat_macro_accuracy_sd"),
            )


if __name__ == "__main__":
    main()
