"""Add an existing-repeat control to the feature-order robustness analysis."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr

from hagf.data import MODALITY_ORDER, load_cohort
from hagf.model import CrossModalFusionClassifier


ROOT = Path(
    os.environ.get("HAGF_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
RUNS = ROOT / "revision_workspace" / "runs"
SUBJECT_ROOT = ROOT / "Methylation" / "SUBJECT"
PILOT = ROOT / "revision_workspace" / "response_evidence" / "feature_order_pilot_20260914"
OUTPUT = ROOT / "revision_workspace" / "response_evidence" / "feature_order_adjusted_20260914"
PERMUTATION_SEED = 20260914

COHORTS = ("CRA001537", "PRJNA929650", "HRA003209")
DISPLAY_NAMES = {
    "CRA001537": "Zhang et al.",
    "PRJNA929650": "Pham et al.",
    "HRA003209": "Bie et al.",
}
BASELINES = {
    "CRA001537": "phase0_full_h100",
    "PRJNA929650": "full_independent",
    "HRA003209": "bie_occlusion_detection_full",
}
COLORS = {
    "FSR": "#4C78A8",
    "EDM": "#E45756",
    "CNV": "#54A24B",
    "Methylation": "#B279A2",
}


def experiment_dir(cohort: str, name: str) -> Path:
    return RUNS / cohort / "detection" / name


def permutation_dir(cohort: str, modality: str) -> Path:
    return experiment_dir(
        cohort,
        f"order_robustness_pilot_{modality}_perm{PERMUTATION_SEED}",
    )


def group_width_scale(feature_count: int, ratio: float = 0.2) -> np.ndarray:
    group_size = max(1, int(feature_count * ratio))
    widths: list[int] = []
    remaining = feature_count
    while remaining:
        width = min(group_size, remaining)
        widths.extend([width] * width)
        remaining -= width
    return np.asarray(widths, dtype=float)


def score_mask(mask: np.ndarray) -> np.ndarray:
    return mask.mean(axis=0) * group_width_scale(mask.shape[1])


def repeat_scores_from_masks(directory: Path) -> dict[str, list[np.ndarray]]:
    scores = {modality: [] for modality in MODALITY_ORDER}
    for repeat in range(1, 6):
        fold_scores = {modality: [] for modality in MODALITY_ORDER}
        for fold in range(1, 11):
            path = directory / f"repeat_{repeat}" / f"fold_{fold}" / "masks.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as archive:
                for branch_index, modality in enumerate(MODALITY_ORDER):
                    fold_scores[modality].append(
                        score_mask(archive[f"branch_{branch_index}_layer_0"])
                    )
        for modality in MODALITY_ORDER:
            scores[modality].append(np.mean(fold_scores[modality], axis=0))
    return scores


def repeat_scores_from_checkpoints(
    directory: Path,
    input_sizes: list[int],
) -> dict[str, list[np.ndarray]]:
    scores = {modality: [] for modality in MODALITY_ORDER}
    model = CrossModalFusionClassifier(
        input_sizes=input_sizes,
        num_layers=2,
        hidden_size=100,
        output_size=2,
        num_masks=4,
        group_ratio=0.2,
        dropout=0.1,
        num_heads=4,
        entmax_alpha=1.1,
    )
    for repeat in range(1, 6):
        fold_scores = {modality: [] for modality in MODALITY_ORDER}
        for fold in range(1, 11):
            path = directory / f"repeat_{repeat}" / f"fold_{fold}" / "best_model.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            for branch_index, layer_index, mask in model.mask_tensors():
                if layer_index == 0:
                    fold_scores[MODALITY_ORDER[branch_index]].append(
                        score_mask(mask.numpy())
                    )
        for modality in MODALITY_ORDER:
            scores[modality].append(np.mean(fold_scores[modality], axis=0))
    return scores


def permuted_score(directory: Path, modality: str) -> np.ndarray:
    branch_index = MODALITY_ORDER.index(modality)
    scores = []
    for fold in range(1, 11):
        path = directory / "repeat_1" / f"fold_{fold}" / "masks.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            scores.append(score_mask(archive[f"branch_{branch_index}_layer_0"]))
    permuted = np.mean(scores, axis=0)
    order = np.load(directory / "feature_permutation.npy")
    restored = np.empty_like(permuted)
    restored[order] = permuted
    return restored


def jaccard(left: np.ndarray, right: np.ndarray, fraction: float = 0.2) -> float:
    count = max(1, int(np.ceil(len(left) * fraction)))
    left_top = set(np.argsort(left)[-count:])
    right_top = set(np.argsort(right)[-count:])
    return len(left_top & right_top) / len(left_top | right_top)


def comparison_record(
    cohort: str,
    modality: str,
    condition: str,
    comparison: str,
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
    return {
        "cohort": cohort,
        "dataset": DISPLAY_NAMES[cohort],
        "modality": modality,
        "condition": condition,
        "comparison": comparison,
        "spearman_rho": float(spearmanr(left, right).statistic),
        "top20_jaccard": jaccard(left, right),
        "feature_count": len(left),
    }


def build_stability_table() -> pd.DataFrame:
    records = []
    for cohort in COHORTS:
        baseline = experiment_dir(cohort, BASELINES[cohort])
        loaded = load_cohort(SUBJECT_ROOT, cohort, "detection", "cross")
        if cohort == "HRA003209":
            original = repeat_scores_from_checkpoints(baseline, loaded.input_sizes)
        else:
            original = repeat_scores_from_masks(baseline)

        for modality in MODALITY_ORDER:
            repeat_rankings = original[modality]
            for left_index, right_index in combinations(range(5), 2):
                records.append(
                    comparison_record(
                        cohort,
                        modality,
                        "Original-order repeats",
                        f"repeat_{left_index + 1}_vs_{right_index + 1}",
                        repeat_rankings[left_index],
                        repeat_rankings[right_index],
                    )
                )
            restored = permuted_score(permutation_dir(cohort, modality), modality)
            for repeat_index, ranking in enumerate(repeat_rankings, start=1):
                records.append(
                    comparison_record(
                        cohort,
                        modality,
                        "Permuted vs original",
                        f"permuted_vs_repeat_{repeat_index}",
                        restored,
                        ranking,
                    )
                )
    return pd.DataFrame(records)


def performance_table() -> pd.DataFrame:
    pilot = pd.read_csv(PILOT / "feature_order_robustness_pilot.csv")
    records = []
    for row in pilot.itertuples(index=False):
        records.append(
            {
                "evaluation": f"{row.dataset} CV",
                "modality": row.modality,
                "delta_auc": row.delta_cv_auc,
            }
        )
        if not np.isnan(row.delta_independent_auc):
            records.append(
                {
                    "evaluation": f"{row.dataset} independent",
                    "modality": row.modality,
                    "delta_auc": row.delta_independent_auc,
                }
            )
    return pd.DataFrame(records)


def draw_ordering_panel(ax) -> None:
    ax.axis("off")
    ax.set_title("A  Feature ordering used before grouping", loc="left", fontsize=13.5, fontweight="bold", pad=8)
    headers = ["Profile", "Stored order", "Interpretation"]
    rows = [
        ["FSR", "22 autosomes x\n3 size intervals", "Structured fragmentomic\nsummaries"],
        ["EDM", "Fixed 4-mer enumeration\nplus MDS", "Computational only;\nno linear biology"],
        ["CNV", "Chromosome, then\n1-Mb coordinate", "Genomic neighborhood"],
        ["Methylation", "Chromosome, then\nregion start", "Genomic neighborhood"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.17, 0.40, 0.43],
        bbox=[0, 0.02, 1, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D4D4D4")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#E9EEF3")
            cell.set_text_props(fontweight="bold", color="#111111")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
        else:
            cell.set_facecolor("white")


def draw_auc_panel(ax, performance: pd.DataFrame) -> None:
    order = [
        "Zhang et al. CV",
        "Pham et al. CV",
        "Pham et al. independent",
        "Bie et al. CV",
        "Bie et al. independent",
    ]
    offsets = dict(zip(MODALITY_ORDER, (-0.24, -0.08, 0.08, 0.24)))
    for modality in MODALITY_ORDER:
        subset = performance[performance["modality"] == modality].set_index("evaluation")
        x = [subset.loc[label, "delta_auc"] for label in order]
        y = [index + offsets[modality] for index in range(len(order))]
        ax.scatter(x, y, s=42, color=COLORS[modality], edgecolor="white", linewidth=0.6, label=modality, zorder=3)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(-0.06, 0.015)
    ax.set_xlabel("AUC change after permutation")
    ax.set_title("B  Predictive sensitivity", loc="left", fontsize=13.5, fontweight="bold", pad=8)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=9, handletextpad=0.4, columnspacing=1.0)


def draw_stability_panel(ax, stability: pd.DataFrame, metric: str, title: str) -> None:
    condition_order = ["Original-order repeats", "Permuted vs original"]
    palette = {"Original-order repeats": "#2A9D8F", "Permuted vs original": "#D97732"}
    sns.boxplot(
        data=stability,
        x="modality",
        y=metric,
        hue="condition",
        order=list(MODALITY_ORDER),
        hue_order=condition_order,
        palette=palette,
        width=0.65,
        fliersize=0,
        linewidth=1,
        ax=ax,
    )
    sns.stripplot(
        data=stability,
        x="modality",
        y=metric,
        hue="condition",
        order=list(MODALITY_ORDER),
        hue_order=condition_order,
        dodge=True,
        palette=palette,
        size=2.7,
        alpha=0.55,
        linewidth=0,
        ax=ax,
        legend=False,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_xlabel("")
    ax.set_ylabel("Spearman rho" if metric == "spearman_rho" else "Top-20% Jaccard index")
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", pad=8)
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    if metric == "spearman_rho":
        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_ylim(-0.35, 0.55)
    else:
        ax.axhline(0.111, color="#666666", linewidth=0.9, linestyle="--")
        ax.text(3.48, 0.118, "random", ha="right", va="bottom", fontsize=8.5, color="#555555")
        ax.set_ylim(0, 0.42)


def plot(performance: pd.DataFrame, stability: pd.DataFrame) -> None:
    sns.set_theme(style="white", font="DejaVu Sans", font_scale=1.0)
    fig = plt.figure(figsize=(12.5, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.93, 1.15])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_ordering_panel(ax_a)
    draw_auc_panel(ax_b, performance)
    draw_stability_panel(ax_c, stability, "spearman_rho", "C  Gate-weight rank stability")
    draw_stability_panel(ax_d, stability, "top20_jaccard", "D  High-weight feature stability")

    handles = [
        Line2D([0], [0], color="#2A9D8F", lw=8, label="Original-order repeats"),
        Line2D([0], [0], color="#D97732", lw=8, label="Permuted vs original"),
    ]
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.18, wspace=0.45, hspace=0.40)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), frameon=False, ncol=2, fontsize=9.5)
    fig.text(
        0.5,
        0.025,
        "One fixed permutation (seed 20260914). Original-order stability uses pairwise comparisons among five existing repeat-averaged rankings; "
        "permuted rankings were inverse-mapped and compared with each original repeat. Original baselines were reused; only the fixed-permutation models were newly trained.",
        ha="center",
        fontsize=8.8,
        color="#404040",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "feature_order_robustness_adjusted.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "feature_order_robustness_adjusted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    performance_path = OUTPUT / "auc_changes.csv"
    stability_path = OUTPUT / "weight_stability.csv"
    if performance_path.is_file() and stability_path.is_file():
        performance = pd.read_csv(performance_path)
        stability = pd.read_csv(stability_path)
    else:
        performance = performance_table()
        stability = build_stability_table()
        performance.to_csv(performance_path, index=False)
        stability.to_csv(stability_path, index=False)
    summary = stability.groupby(["modality", "condition"])[["spearman_rho", "top20_jaccard"]].agg(["median", "min", "max"])
    summary.to_csv(OUTPUT / "weight_stability_summary.csv")
    with (OUTPUT / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "permutation_seed": PERMUTATION_SEED,
                "new_training": False,
                "baseline_repeats": 5,
                "same_order_comparisons_per_cohort_modality": 10,
                "permuted_comparisons_per_cohort_modality": 5,
                "gate_layer": 0,
                "gate_masks_averaged": 4,
                "fold_models_averaged_per_ranking": 10,
                "top_feature_fraction": 0.2,
            },
            handle,
            indent=2,
        )
    plot(performance, stability)
    print(summary.to_string())
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
