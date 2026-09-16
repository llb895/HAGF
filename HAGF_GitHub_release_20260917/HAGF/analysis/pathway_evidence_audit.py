"""Audit the provenance and significance fields of existing HCC pathway results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


CLAIMED = {
    "CNV": [
        "Hepatocellular carcinoma",
        "MAPK signaling pathway",
        "Rap1 signaling pathway",
        "Breast cancer",
        "Chemokine signaling pathway",
        "Apelin signaling pathway",
        "Estrogen signaling pathway",
        "Hormone signaling",
    ],
    "Methylation": [
        "MAPK signaling pathway",
        "Hippo signaling pathway",
        "VEGF signaling pathway",
        "Prolactin signaling pathway",
        "Cholesterol metabolism",
        "Arachidonic acid metabolism",
        "Peroxisome",
        "Fatty acid elongation",
        "Valine, leucine and isoleucine degradation",
    ],
}


def mask_audit(path: Path) -> dict:
    frame = pd.read_csv(path)
    return {
        "path": str(path),
        "rows": len(frame),
        "feature_columns": len(frame.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cra = args.subject_root / "CRA001537"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "CNV": cra / "KEGG" / "copy_numberl.xlsx",
        "Methylation": cra / "KEGG" / "methy.all.xlsx",
    }
    claimed_tables = []
    workbook_summary = {}
    for modality, path in files.items():
        frame = pd.read_excel(path)
        frame.insert(0, "Modality", modality)
        frame.to_csv(args.output_dir / f"existing_{modality.lower()}_all_pathways.csv", index=False)
        claimed = frame.loc[frame["Description"].isin(CLAIMED[modality])].copy()
        claimed["FDR_0.05"] = claimed["p.adjust"] < 0.05
        claimed_tables.append(claimed)
        missing = sorted(set(CLAIMED[modality]) - set(claimed["Description"]))
        workbook_summary[modality] = {
            "workbook": str(path),
            "pathways_total": len(frame),
            "pathways_adjusted_p_below_0.05": int((frame["p.adjust"] < 0.05).sum()),
            "claimed_pathways_found": len(claimed),
            "claimed_pathways_adjusted_p_below_0.05": int(
                (claimed["p.adjust"] < 0.05).sum()
            ),
            "claimed_pathways_missing": missing,
        }

    claimed_table = pd.concat(claimed_tables, ignore_index=True)
    claimed_table.to_csv(args.output_dir / "existing_claimed_pathway_statistics.csv", index=False)

    main_dimensions = {
        "CNV": pd.read_csv(
            args.subject_root / "data" / "dataset" / "CRA001537_cross" / "CopyNumber.csv",
            nrows=1,
        ).shape[1]
        - 2,
        "Methylation": pd.read_csv(
            args.subject_root / "data" / "dataset" / "CRA001537_cross" / "methy_selected.csv",
            nrows=1,
        ).shape[1]
        - 2,
    }
    auxiliary_methylation_dimension = (
        pd.read_csv(
            args.subject_root / "data" / "dataset" / "CRA001537_cross" / "methy_new_stand.csv",
            nrows=1,
        ).shape[1]
        - 2
    )
    top_file = cra / "bio20260307" / "top20_indices_Methy.txt"
    top_file_regions = [line.strip() for line in top_file.read_text().splitlines() if line.strip()]

    summary = {
        "existing_workbooks": workbook_summary,
        "main_model_feature_dimensions": main_dimensions,
        "strict_top_20_percent_counts_main_model_ceiling": {
            name: math.ceil(dimension * 0.2)
            for name, dimension in main_dimensions.items()
        },
        "auxiliary_full_resolution_methylation_dimension": auxiliary_methylation_dimension,
        "strict_top_20_percent_count_auxiliary_methylation_ceiling": math.ceil(
            auxiliary_methylation_dimension * 0.2
        ),
        "existing_top20_methylation_file_region_count": len(top_file_regions),
        "existing_mask_files": [
            mask_audit(cra / "plot" / "CopyNumber_analyse" / f"mask_{index}.csv")
            for index in (1, 2)
        ]
        + [
            mask_audit(cra / "plot" / "Methylation_Map" / f"mask_{index}.csv")
            for index in (1, 2)
        ],
        "decision": (
            "Do not reuse the old top-feature selection as final reviewer evidence. "
            "Regenerate feature weights from the frozen revision model and apply one "
            "explicit top-20-percent rule before region-to-gene mapping and enrichment."
        ),
    }
    with (args.output_dir / "pathway_provenance_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(claimed_table[["Modality", "ID", "Description", "pvalue", "p.adjust", "qvalue", "Count"]].to_string(index=False))


if __name__ == "__main__":
    main()
