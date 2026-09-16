"""Recompute Figure 4 group comparisons with sample-level BH-FDR correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def load_predictions(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no prediction CSV files in {path}")
    frames = []
    for csv_path in files:
        frame = pd.read_csv(csv_path)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        if not {"id", "label", "probability"}.issubset(columns):
            continue
        frames.append(
            frame[[columns["id"], columns["label"], columns["probability"]]].rename(
                columns={
                    columns["id"]: "ID",
                    columns["label"]: "Label",
                    columns["probability"]: "Probability",
                }
            )
        )
    if not frames:
        raise ValueError(f"no compatible prediction files in {path}")
    stacked = pd.concat(frames, ignore_index=True)
    consistency = stacked.groupby("ID")["Label"].nunique()
    if consistency.max() != 1:
        raise ValueError(f"inconsistent labels across repeated predictions in {path}")
    return (
        stacked.groupby("ID", as_index=False)
        .agg(Label=("Label", "first"), Probability=("Probability", "mean"), repeats=("ID", "size"))
    )


def bh_panel(
    frame: pd.DataFrame,
    panel: str,
    group_column: str,
    reference: str,
    comparisons: list[str],
) -> list[dict]:
    reference_values = frame.loc[frame[group_column] == reference, "Probability"].dropna()
    raw = []
    for comparison in comparisons:
        values = frame.loc[frame[group_column] == comparison, "Probability"].dropna()
        if len(reference_values) == 0 or len(values) == 0:
            continue
        statistic, p_value = mannwhitneyu(
            reference_values, values, alternative="two-sided", method="auto"
        )
        raw.append(
            {
                "panel": panel,
                "reference": reference,
                "comparison": comparison,
                "n_reference": int(len(reference_values)),
                "n_comparison": int(len(values)),
                "median_reference": float(np.median(reference_values)),
                "median_comparison": float(np.median(values)),
                "mann_whitney_u": float(statistic),
                "p_value": float(p_value),
            }
        )
    if not raw:
        return raw
    rejected, adjusted, _, _ = multipletests(
        [item["p_value"] for item in raw], alpha=0.05, method="fdr_bh"
    )
    for item, reject, q_value in zip(raw, rejected, adjusted):
        item["fdr_bh_q_value"] = float(q_value)
        item["significant_at_fdr_0.05"] = bool(reject)
    return raw


def hra_results(subject_root: Path) -> list[dict]:
    metadata_path = subject_root.parent / "HRA003209" / "HRA003209_info.xlsx"
    metadata = pd.read_excel(metadata_path, header=None, usecols=[0, 2, 3, 5])
    metadata.columns = ["ID", "CancerType", "Partition", "Stage"]
    metadata["ID"] = metadata["ID"].astype(str)
    metadata["CancerType"] = metadata["CancerType"].astype(str)
    metadata.loc[metadata["CancerType"].eq("healthy"), "Stage"] = "healthy"
    metadata["Stage"] = metadata["Stage"].astype(str).str.strip()

    cohort_root = subject_root / "HRA003209" / "plot"
    settings = (
        ("cross", "Training", cohort_root / "HRA003209_5_10folds_cross"),
        ("validation", "Test", cohort_root / "HRA003209_5_10folds_val"),
    )
    output = []
    for split, partition, prediction_path in settings:
        predictions = load_predictions(prediction_path)
        merged = predictions.merge(
            metadata.loc[metadata["Partition"] == partition], on="ID", how="left", validate="one_to_one"
        )
        if merged[["CancerType", "Stage"]].isna().any().any():
            raise ValueError(f"missing HRA metadata after merging {split}")
        cancer_types = sorted(
            value for value in merged["CancerType"].unique() if value != "healthy"
        )
        stages = [
            value
            for value in ("I", "II", "III", "IV")
            if value in set(merged["Stage"])
        ]
        output.extend(
            bh_panel(
                merged,
                f"HRA003209_{split}_cancer_type",
                "CancerType",
                "healthy",
                cancer_types,
            )
        )
        output.extend(
            bh_panel(
                merged,
                f"HRA003209_{split}_stage",
                "Stage",
                "healthy",
                stages,
            )
        )
    return output


def cra_results(subject_root: Path) -> list[dict]:
    metadata = pd.read_excel(subject_root / "CRA001537" / "CRA001537.xlsx")
    metadata.columns = ["ID", "Description"]
    description = metadata["Description"].astype(str).str.lower()
    metadata["DiseaseGroup"] = np.select(
        [
            description.str.contains("hepatitis"),
            description.str.contains("cirrhosis"),
            description.str.contains("hcc"),
        ],
        ["hepatitis", "cirrhosis", "HCC"],
        default="healthy",
    )
    predictions = load_predictions(
        subject_root / "CRA001537" / "plot" / "CRA001537_5_10folds"
    )
    merged = predictions.merge(metadata[["ID", "DiseaseGroup"]], on="ID", how="left", validate="one_to_one")
    if merged["DiseaseGroup"].isna().any():
        raise ValueError("missing CRA metadata after merging")
    return bh_panel(
        merged,
        "CRA001537_disease_group",
        "DiseaseGroup",
        "HCC",
        ["healthy", "hepatitis", "cirrhosis"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparisons = cra_results(args.subject_root) + hra_results(args.subject_root)
    report = {
        "unit_of_analysis": "unique sample",
        "repeated_prediction_handling": "mean probability across five repeated CV predictions",
        "test": "two-sided Mann-Whitney U",
        "multiple_testing": "Benjamini-Hochberg within each figure panel",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(comparisons).to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"FDR_COMPLETE {args.output}")
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
