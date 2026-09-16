"""Summarize and plot retrained leave-one-modality-out cancer detection runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(
    os.environ.get("HAGF_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
RUNS = ROOT / "revision_workspace" / "runs"
OUT = ROOT / "revision_workspace" / "response_evidence" / "reviewer4_lomo_detection"

MODALITIES = ["FSR", "EDM", "CNV", "Methylation"]
CONDITIONS = ["Full HAGF", "-FSR", "-EDM", "-CNV", "-Methylation"]
COHORTS = ["CRA001537", "PRJNA929650", "HRA003209"]
DISPLAY_NAMES = {
    "CRA001537": "Zhang et al. dataset",
    "PRJNA929650": "Pham et al. dataset",
    "HRA003209": "Bie et al. dataset",
}
FULL_RUNS = {
    "CRA001537": RUNS / "CRA001537" / "detection" / "phase0_full_h100",
    "PRJNA929650": RUNS / "PRJNA929650" / "detection" / "full_independent",
    "HRA003209": RUNS / "HRA003209" / "detection" / "full_independent",
}
HISTORICAL_CV = {
    "CRA001537": ROOT
    / "Methylation/SUBJECT/CRA001537/plot/CRA001537_5_10folds",
    "PRJNA929650": ROOT
    / "Methylation/SUBJECT/PRJNA929650/plot/PRJNA929650_5_10folds_cross",
    "HRA003209": ROOT
    / "Methylation/SUBJECT/HRA003209/plot/HRA003209_5_10folds_cross",
}
HISTORICAL_INDEPENDENT = {
    "PRJNA929650": ROOT
    / "Methylation/SUBJECT/PRJNA929650/plot/PRJNA929650_5_10folds_val/model_val.csv",
    "HRA003209": ROOT
    / "Methylation/SUBJECT/HRA003209/plot/HRA003209_5_10folds_val/model_val.csv",
}
COLORS = {"CV": "#16877A", "Independent": "#C8752D"}
MARKERS = {"CV": "o", "Independent": "s"}


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame if column.startswith("Probability_")]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[1]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(
    run: Path, repeat: int, independent: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filename = "independent_predictions.csv" if independent else "predictions.csv"
    frames = []
    for fold in range(1, 11):
        path = run / f"repeat_{repeat}" / f"fold_{fold}" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if not frame["ID"].is_unique or frame["Label"].isna().any():
            raise AssertionError(f"Invalid prediction rows: {path}")
        frames.append(frame)

    if independent:
        reference = frames[0].sort_values("ID").reset_index(drop=True)
        columns = probability_columns(reference)
        probabilities = []
        for frame in frames:
            aligned = frame.set_index("ID").loc[reference["ID"]]
            if not np.array_equal(
                aligned["Label"].to_numpy(), reference["Label"].to_numpy()
            ):
                raise AssertionError("Independent labels differ across folds")
            probabilities.append(aligned[columns].to_numpy())
        return (
            reference["ID"].astype(str).to_numpy(),
            reference["Label"].to_numpy(),
            np.mean(probabilities, axis=0),
        )

    joined = pd.concat(frames, ignore_index=True)
    if not joined["ID"].is_unique:
        raise AssertionError("Cross-validation IDs are not unique within a repeat")
    columns = probability_columns(joined)
    return (
        joined["ID"].astype(str).to_numpy(),
        joined["Label"].to_numpy(),
        joined[columns].to_numpy(),
    )


def load_historical_full(
    cohort: str, repeat: int, independent: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if independent:
        frame = pd.read_csv(HISTORICAL_INDEPENDENT[cohort])
    else:
        frames = [
            pd.read_csv(HISTORICAL_CV[cohort] / f"{repeat}_{fold}_fold.csv")
            for fold in range(1, 11)
        ]
        frame = pd.concat(frames, ignore_index=True)
    if not frame["ID"].is_unique:
        raise AssertionError("Historical prediction IDs are not unique")
    return (
        frame["ID"].astype(str).to_numpy(),
        frame["Label"].to_numpy(),
        frame["Probability"].to_numpy(),
    )


def align_to_reference(
    reference_ids: np.ndarray,
    reference_labels: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    lookup = {identifier: index for index, identifier in enumerate(ids)}
    if set(reference_ids) != set(ids):
        raise AssertionError("Full and LOMO predictions contain different sample IDs")
    order = np.asarray([lookup[identifier] for identifier in reference_ids])
    if not np.array_equal(reference_labels, labels[order]):
        raise AssertionError("Full and LOMO labels differ after ID alignment")
    return probabilities[order]


def assert_matching_splits(full: Path, dropped: Path) -> None:
    for repeat in range(1, 6):
        for fold in range(1, 11):
            full_path = full / f"repeat_{repeat}" / f"fold_{fold}" / "split.json"
            drop_path = dropped / f"repeat_{repeat}" / f"fold_{fold}" / "split.json"
            full_split = json.loads(full_path.read_text())
            drop_split = json.loads(drop_path.read_text())
            if full_split != drop_split:
                raise AssertionError(
                    f"Split mismatch: {full_path} versus {drop_path}"
                )


def build_records() -> pd.DataFrame:
    records = []
    for cohort in COHORTS:
        full = FULL_RUNS[cohort]
        dropped_runs = {
            modality: RUNS
            / cohort
            / "detection"
            / f"lomout_detection_without_{modality}"
            for modality in MODALITIES
        }
        for dropped in dropped_runs.values():
            assert_matching_splits(full, dropped)

        evaluations = [("CV", False)]
        if cohort != "CRA001537":
            evaluations.append(("Independent", True))
        for repeat in range(1, 6):
            for evaluation, independent in evaluations:
                ids, labels, _ = load_predictions(
                    full, repeat, independent
                )
                historical_ids, historical_labels, historical_probabilities = (
                    load_historical_full(cohort, repeat, independent)
                )
                historical_probabilities = align_to_reference(
                    ids,
                    labels,
                    historical_ids,
                    historical_labels,
                    historical_probabilities,
                )
                full_auc = float(roc_auc_score(labels, historical_probabilities))
                records.append(
                    {
                        "cohort": cohort,
                        "evaluation": evaluation,
                        "repeat": repeat,
                        "condition": "Full HAGF",
                        "removed_modality": "",
                        "auc": full_auc,
                        "full_auc": full_auc,
                        "delta_auc": 0.0,
                    }
                )
                for modality, dropped in dropped_runs.items():
                    drop_ids, drop_labels, drop_probabilities = load_predictions(
                        dropped, repeat, independent
                    )
                    drop_probabilities = align_to_reference(
                        ids,
                        labels,
                        drop_ids,
                        drop_labels,
                        drop_probabilities,
                    )
                    drop_auc = float(
                        roc_auc_score(labels, drop_probabilities[:, 1])
                    )
                    records.append(
                        {
                            "cohort": cohort,
                            "evaluation": evaluation,
                            "repeat": repeat,
                            "condition": f"-{modality}",
                            "removed_modality": modality,
                            "auc": drop_auc,
                            "full_auc": full_auc,
                            "delta_auc": full_auc - drop_auc,
                        }
                    )
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    summary = (
        records.groupby(
            ["cohort", "evaluation", "condition", "removed_modality"],
            dropna=False,
            sort=False,
        )
        .agg(
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
            mean_delta_auc=("delta_auc", "mean"),
            sd_delta_auc=("delta_auc", "std"),
            n_repeats=("repeat", "size"),
        )
        .reset_index()
    )
    summary["condition"] = pd.Categorical(
        summary["condition"], categories=CONDITIONS, ordered=True
    )
    return summary.sort_values(["cohort", "evaluation", "condition"])


def plot(records: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.35), sharey=True)
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = {"CV": -0.10, "Independent": 0.10}
    minimum = float(records["auc"].min())
    lower = max(0.0, np.floor((minimum - 0.04) * 20.0) / 20.0)

    for panel, (axis, cohort) in enumerate(zip(axes, COHORTS)):
        cohort_records = records.loc[records["cohort"].eq(cohort)]
        evaluations = [
            evaluation
            for evaluation in ("CV", "Independent")
            if cohort_records["evaluation"].eq(evaluation).any()
        ]
        for evaluation in evaluations:
            evaluation_records = cohort_records.loc[
                cohort_records["evaluation"].eq(evaluation)
            ]
            offset = 0.0 if len(evaluations) == 1 else offsets[evaluation]
            color = COLORS[evaluation]
            marker = MARKERS[evaluation]
            matrix = (
                evaluation_records.pivot(
                    index="repeat", columns="condition", values="auc"
                )
                .loc[range(1, 6), CONDITIONS]
                .to_numpy()
            )
            for repeat_values in matrix:
                axis.plot(
                    x + offset,
                    repeat_values,
                    color=color,
                    linewidth=0.75,
                    alpha=0.18,
                    zorder=1,
                )
            for condition_index, condition in enumerate(CONDITIONS):
                values = matrix[:, condition_index]
                jitter = np.linspace(-0.025, 0.025, len(values))
                axis.scatter(
                    np.full(len(values), x[condition_index] + offset) + jitter,
                    values,
                    marker=marker,
                    s=22,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    zorder=3,
                )
                axis.errorbar(
                    x[condition_index] + offset,
                    float(values.mean()),
                    yerr=float(values.std(ddof=1)),
                    fmt=marker,
                    markersize=6.2,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    elinewidth=1.4,
                    capsize=3.5,
                    label=evaluation if condition_index == 0 else None,
                    zorder=4,
                )
        tick_labels = ["Full HAGF\n(archived)", *CONDITIONS[1:]]
        axis.set_xticks(x, tick_labels, rotation=20, ha="right")
        axis.set_xlim(-0.50, len(CONDITIONS) - 0.50)
        axis.set_ylim(lower, 1.01)
        axis.grid(axis="y", color="#D6D6D6", linewidth=0.7, alpha=0.55)
        axis.set_title(
            f"{'ABC'[panel]}  {DISPLAY_NAMES[cohort]}",
            loc="left",
            fontweight="bold",
            pad=9,
        )
        if panel == 0:
            axis.set_ylabel("ROC AUC")

    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.005),
        frameon=False,
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    figure.subplots_adjust(
        left=0.065, right=0.99, bottom=0.21, top=0.82, wspace=0.25
    )
    figure.savefig(OUT / "revise_4_1.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(
        OUT / "revise_4_1_preview.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUT / "Supplementary_Figure_LOMO.tiff",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    expected_rows = 5 * 5 + 10 * 5 + 10 * 5
    if len(records) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(records)}")
    if not np.isfinite(records[["auc", "full_auc", "delta_auc"]]).all().all():
        raise AssertionError("Non-finite AUC values")
    records.to_csv(OUT / "lomo_detection_results.csv", index=False)
    summary = summarize(records)
    if not (summary["n_repeats"] == 5).all():
        raise AssertionError("Every plotted estimate must contain five repeats")
    summary.to_csv(OUT / "lomo_detection_summary.csv", index=False)
    plot(records)

    audit = {
        "analysis": "archived full HAGF versus retrained leave-one-modality-out variants",
        "full_models_retrained": False,
        "lomo_models_retrained": True,
        "paired_splits_verified": True,
        "full_baseline_provenance": "historical prediction CSVs transferred from the original server",
        "comparison_scope": "identical held-out sample partitions; separate training batches",
        "cohorts": COHORTS,
        "conditions": CONDITIONS,
        "rows": int(len(records)),
        "repeats_per_setting": 5,
        "folds_per_repeat": 10,
        "output_sha256": sha256(OUT / "revise_4_1.pdf"),
    }
    (OUT / "verification.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    print("REVIEWER4_LOMO_FIGURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
