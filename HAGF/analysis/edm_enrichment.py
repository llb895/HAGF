"""Recompute CRA001537 end-motif enrichment and heatmap normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests


def disease_group(description: str) -> str:
    value = str(description).lower()
    if "hepatitis" in value:
        return "hepatitis"
    if "cirrhosis" in value:
        return "cirrhosis"
    if "hcc" in value:
        return "HCC"
    return "healthy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    matrix_path = (
        args.subject_root / "data" / "dataset" / "CRA001537_cross" / "end_motifs.csv"
    )
    metadata_path = args.subject_root / "CRA001537" / "CRA001537.xlsx"
    frame = pd.read_csv(matrix_path)
    metadata = pd.read_excel(metadata_path)
    metadata.columns = ["ID", "Description"]
    metadata["DiseaseGroup"] = metadata["Description"].map(disease_group)
    merged = frame.merge(metadata[["ID", "DiseaseGroup"]], on="ID", how="left", validate="one_to_one")
    if merged["DiseaseGroup"].isna().any():
        raise ValueError("missing disease group after joining EDM matrix to metadata")

    motif_columns = [
        column
        for column in frame.columns
        if column not in {"ID", "class", "MDS"} and len(str(column)) == 4
    ]
    if len(motif_columns) != 256:
        raise ValueError(f"expected 256 tetranucleotide motifs, found {len(motif_columns)}")

    hcc = merged["DiseaseGroup"].eq("HCC")
    records = []
    for motif in motif_columns:
        hcc_values = merged.loc[hcc, motif].to_numpy(dtype=float)
        non_hcc_values = merged.loc[~hcc, motif].to_numpy(dtype=float)
        u_statistic, p_value = mannwhitneyu(
            hcc_values, non_hcc_values, alternative="two-sided", method="auto"
        )
        t_statistic, welch_p_value = ttest_ind(
            hcc_values, non_hcc_values, equal_var=False
        )
        mean_hcc = float(np.mean(hcc_values))
        mean_non_hcc = float(np.mean(non_hcc_values))
        records.append(
            {
                "motif": motif,
                "mean_hcc": mean_hcc,
                "mean_non_hcc": mean_non_hcc,
                "median_hcc": float(np.median(hcc_values)),
                "median_non_hcc": float(np.median(non_hcc_values)),
                "mean_difference_hcc_minus_non_hcc": mean_hcc - mean_non_hcc,
                "rank_biserial_hcc_over_non_hcc": float(
                    2.0 * u_statistic / (len(hcc_values) * len(non_hcc_values)) - 1.0
                ),
                "mann_whitney_u": float(u_statistic),
                "p_value": float(p_value),
                "welch_t": float(t_statistic),
                "welch_p_value_secondary": float(welch_p_value),
            }
        )

    statistics = pd.DataFrame(records)
    statistics["fdr_bh_q_value"] = multipletests(
        statistics["p_value"], alpha=0.05, method="fdr_bh"
    )[1]
    statistics["enrichment_direction"] = np.where(
        statistics["mean_hcc"] > statistics["mean_non_hcc"],
        "HCC-enriched",
        "non-HCC-enriched",
    )
    statistics["significant_fdr_0.05"] = statistics["fdr_bh_q_value"] < 0.05
    statistics = statistics.sort_values(
        ["significant_fdr_0.05", "enrichment_direction", "fdr_bh_q_value", "mean_difference_hcc_minus_non_hcc"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    selected = statistics.loc[statistics["significant_fdr_0.05"]].copy()
    selected["direction_order"] = selected["enrichment_direction"].map(
        {"HCC-enriched": 0, "non-HCC-enriched": 1}
    )
    selected["effect_magnitude"] = selected[
        "mean_difference_hcc_minus_non_hcc"
    ].abs()
    selected = selected.sort_values(
        ["direction_order", "effect_magnitude"],
        ascending=[True, False],
    )

    values = merged.set_index("ID")[selected["motif"].tolist()].T.astype(float)
    row_means = values.mean(axis=1)
    row_sd = values.std(axis=1, ddof=1).replace(0.0, np.nan)
    heatmap_z = values.sub(row_means, axis=0).div(row_sd, axis=0).fillna(0.0)
    sample_order = (
        merged.assign(_hcc_order=np.where(hcc, 0, 1))
        .sort_values(["_hcc_order", "DiseaseGroup", "ID"])["ID"]
        .tolist()
    )
    heatmap_z = heatmap_z.loc[selected["motif"], sample_order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(args.output_dir / "edm_all_256_statistics.csv", index=False)
    selected.drop(columns=["direction_order", "effect_magnitude"]).to_csv(
        args.output_dir / "edm_fdr_significant_motifs.csv", index=False
    )
    heatmap_z.to_csv(args.output_dir / "edm_heatmap_row_zscores.csv")
    sample_annotations = merged[["ID", "DiseaseGroup"]].copy()
    sample_annotations.to_csv(args.output_dir / "edm_heatmap_sample_annotations.csv", index=False)

    counts = selected["enrichment_direction"].value_counts().to_dict()
    input_values = frame[motif_columns].to_numpy(dtype=float)
    summary = {
        "samples": {
            "HCC": int(hcc.sum()),
            "non-HCC": int((~hcc).sum()),
        },
        "motifs_tested": len(motif_columns),
        "primary_test": "two-sided Mann-Whitney U",
        "multiple_testing": "Benjamini-Hochberg across 256 motifs",
        "significance_threshold": "FDR q < 0.05",
        "input_scale_audit": {
            "minimum": float(np.min(input_values)),
            "maximum": float(np.max(input_values)),
            "median_within_sample_mean": float(np.median(np.mean(input_values, axis=1))),
            "median_within_sample_sd": float(np.median(np.std(input_values, axis=1, ddof=1))),
            "interpretation": "pre-standardized model-input motif values, not raw non-negative relative abundances",
        },
        "direction_definition": "larger arithmetic mean standardized motif value",
        "effect_sizes": "mean difference and rank-biserial correlation",
        "heatmap_normalization": "row-wise z-score across all 60 samples applied to the pre-standardized model-input motif matrix",
        "heatmap_row_order": "HCC-enriched followed by non-HCC-enriched; descending absolute HCC-minus-non-HCC mean difference within direction",
        "significant_total": int(len(selected)),
        "HCC_enriched": int(counts.get("HCC-enriched", 0)),
        "non_HCC_enriched": int(counts.get("non-HCC-enriched", 0)),
    }
    with (args.output_dir / "edm_enrichment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
