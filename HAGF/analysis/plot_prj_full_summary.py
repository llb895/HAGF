#!/usr/bin/env python3
"""Create publication-ready PRJNA929650 baseline summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


CV_COLOR = "#00796B"
INDEPENDENT_COLOR = "#D97706"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#202124"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cv_summary = load_json(args.run_dir / "summary.json")
    independent_summary = load_json(args.run_dir / "independent_summary.json")
    cv_repeats = pd.read_csv(args.run_dir / "repeat_metrics.csv")
    independent_repeats = pd.read_csv(
        args.run_dir / "independent_repeat_metrics.csv"
    )
    cv_predictions = pd.read_csv(args.run_dir / "sample_averaged_predictions.csv")
    independent_predictions = pd.read_csv(
        args.run_dir / "independent_sample_averaged_predictions.csv"
    )

    repeat_table = cv_repeats[["repeat", "auc"]].rename(
        columns={"auc": "cv_auc"}
    ).merge(
        independent_repeats[["repeat", "ensemble_auc"]].rename(
            columns={"ensemble_auc": "independent_ensemble_auc"}
        ),
        on="repeat",
        validate="one_to_one",
    )
    repeat_table.to_csv(args.output_dir / "PRJNA929650_repeat_auc.csv", index=False)

    summary_table = pd.DataFrame(
        [
            {
                "analysis": "Repeated 10-fold cross-validation",
                "reporting_role": "Primary",
                "auc": cv_summary["repeat_auc_mean"],
                "sd": cv_summary["repeat_auc_sample_sd"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            },
            {
                "analysis": "Independent validation; 10-fold ensemble per repeat",
                "reporting_role": "Primary",
                "auc": independent_summary["repeat_ensemble_auc_mean"],
                "sd": independent_summary["repeat_ensemble_auc_sample_sd"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            },
            {
                "analysis": "Cross-validation; predictions averaged across repeats",
                "reporting_role": "Descriptive ensemble",
                "auc": cv_summary["sample_averaged_auc"],
                "sd": np.nan,
                "ci_lower": cv_summary["sample_averaged_auc_bootstrap_95_ci"][0],
                "ci_upper": cv_summary["sample_averaged_auc_bootstrap_95_ci"][1],
            },
            {
                "analysis": "Independent validation; all-model ensemble",
                "reporting_role": "Descriptive ensemble",
                "auc": independent_summary["all_model_ensemble_auc"],
                "sd": np.nan,
                "ci_lower": independent_summary[
                    "all_model_ensemble_auc_bootstrap_95_ci"
                ][0],
                "ci_upper": independent_summary[
                    "all_model_ensemble_auc_bootstrap_95_ci"
                ][1],
            },
        ]
    )
    summary_table.to_csv(args.output_dir / "PRJNA929650_auc_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.75), constrained_layout=True)

    ax = axes[0]
    repeats = repeat_table["repeat"].to_numpy()
    cv_values = repeat_table["cv_auc"].to_numpy()
    independent_values = repeat_table["independent_ensemble_auc"].to_numpy()
    for values, color, marker, label in (
        (
            cv_values,
            CV_COLOR,
            "o",
            f"Repeated CV: {cv_values.mean():.3f} +/- {cv_values.std(ddof=1):.3f}",
        ),
        (
            independent_values,
            INDEPENDENT_COLOR,
            "s",
            "Independent validation: "
            f"{independent_values.mean():.3f} +/- {independent_values.std(ddof=1):.3f}",
        ),
    ):
        mean = values.mean()
        sd = values.std(ddof=1)
        ax.fill_between(
            [0.8, 5.2], mean - sd, mean + sd, color=color, alpha=0.10, linewidth=0
        )
        ax.axhline(mean, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.plot(
            repeats,
            values,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.5,
            label=label,
        )
    ax.set_title("A. Repeat-level performance", loc="left", fontweight="bold")
    ax.set_xlabel("Repeat")
    ax.set_ylabel("AUC")
    ax.set_xticks(repeats)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.65)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    for predictions, color, linestyle, name, ci in (
        (
            cv_predictions,
            CV_COLOR,
            "-",
            "CV repeated ensemble",
            cv_summary["sample_averaged_auc_bootstrap_95_ci"],
        ),
        (
            independent_predictions,
            INDEPENDENT_COLOR,
            "-",
            "Independent 50-model ensemble",
            independent_summary["all_model_ensemble_auc_bootstrap_95_ci"],
        ),
    ):
        fpr, tpr, _ = roc_curve(predictions["Label"], predictions["Probability_1"])
        curve_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{name}\nAUC {curve_auc:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle=":", linewidth=1.2)
    ax.set_title("B. Ensemble ROC curves", loc="left", fontweight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.55)
    ax.legend(loc="lower right", frameon=False)

    output_stem = args.output_dir / "PRJNA929650_full_baseline_summary"
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    notes = (
        "PRJNA929650 full-model baseline summary\n\n"
        "Panel A reports the primary repeat-level statistics. Cross-validation AUC "
        "is calculated once per complete outer 10-fold repeat. Independent-validation "
        "AUC is calculated after averaging the predictions of the 10 fold-specific "
        "models within each repeat. Shaded bands show mean +/- one sample SD.\n\n"
        "Panel B is a descriptive ensemble analysis. Cross-validation probabilities "
        "are averaged across five repeated out-of-fold predictions per sample. "
        "Independent-validation probabilities are averaged across all 50 trained "
        "models. Confidence intervals are sample-stratified bootstrap intervals "
        "with 5,000 iterations. These ensemble AUCs should not replace the repeat-level "
        "mean +/- SD in the main statistical report.\n"
    )
    (args.output_dir / "PRJNA929650_figure_notes.txt").write_text(
        notes, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
