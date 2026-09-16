"""Build the reviewer figure using frozen-model modality occlusion."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hagf.data import load_cohort
from hagf.model import AblationConfig, CrossModalFusionClassifier
from hagf.runner import standardize_fold

ROOT = Path(
    os.environ.get("HAGF_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
RUNS = ROOT / "revision_workspace/runs"
OUT = ROOT / "revision_workspace/response_evidence/revise_3_1_occlusion"
FINAL = ROOT / "revision_workspace/response_evidence/revise_3_1/revise_3_1.pdf"
OUT.mkdir(parents=True, exist_ok=True)
FINAL.parent.mkdir(parents=True, exist_ok=True)

MODALITIES = ["FSR", "EDM", "CNV", "Methylation"]
CANCERS = ["BRCA", "COREAD", "ESCA", "STAD", "LIHC", "NSCLC", "PACA"]
DETECTION_RUNS = {
    "CRA001537": "phase0_full_h100",
    "PRJNA929650": "full_independent",
    "HRA003209": "bie_occlusion_detection_full",
}


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame if column.startswith("Probability_")]
    return sorted(columns, key=lambda column: int(column.rsplit("_", 1)[1]))


def infer_conditions(model, matrices, device, batch_size=256):
    """Return baseline plus four zero-embedding modality-occlusion predictions."""
    outputs = [[] for _ in range(1 + len(MODALITIES))]
    for start in range(0, len(matrices[0]), batch_size):
        tensors = [
            torch.as_tensor(matrix[start : start + batch_size], device=device)
            for matrix in matrices
        ]
        with torch.inference_mode():
            embeddings = [branch(x) for branch, x in zip(model.branches, tensors)]
            base = torch.softmax(
                model.fusion_classifier(torch.cat(embeddings, dim=1)), dim=1
            )
            outputs[0].append(base.cpu().numpy())
            for modality_index in range(len(MODALITIES)):
                changed = list(embeddings)
                changed[modality_index] = torch.zeros_like(changed[modality_index])
                probability = torch.softmax(
                    model.fusion_classifier(torch.cat(changed, dim=1)), dim=1
                )
                outputs[1 + modality_index].append(probability.cpu().numpy())
    return np.stack([np.concatenate(parts) for parts in outputs])


def run_records(cohort_name, task, experiment, device):
    folder = RUNS / cohort_name / task / experiment
    config = json.loads((folder / "run_config.json").read_text())
    cohort = load_cohort(Path(config["subject_root"]), cohort_name, task=task, split="cross")
    independent = load_cohort(
        Path(config["subject_root"]), cohort_name, task=task, split="validation"
    ) if config.get("evaluate_independent", False) else None
    assert list(cohort.feature_names) == MODALITIES
    if independent is not None:
        assert list(independent.feature_names) == MODALITIES
        assert not set(cohort.ids) & set(independent.ids)

    id_lookup = {str(identifier): index for index, identifier in enumerate(cohort.ids)}
    output_size = 2 if task == "detection" else 8
    records = []
    audits = []
    for repeat in range(1, 6):
        cv_probabilities = np.full(
            (1 + len(MODALITIES), len(cohort.ids), output_size), np.nan
        )
        seen = np.zeros(len(cohort.ids), dtype=int)
        independent_sum = None if independent is None else np.zeros(
            (1 + len(MODALITIES), len(independent.ids), output_size)
        )
        for fold in range(1, 11):
            fold_dir = folder / f"repeat_{repeat}" / f"fold_{fold}"
            required = [
                fold_dir / "best_model.pt",
                fold_dir / "split.json",
                fold_dir / "predictions.csv",
            ]
            if independent is not None:
                required.append(fold_dir / "independent_predictions.csv")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing completed-fold artifacts: " + ", ".join(missing))

            split = json.loads((fold_dir / "split.json").read_text())
            train, validation, test = [
                np.asarray([id_lookup[str(identifier)] for identifier in split[key]])
                for key in ("train_ids", "validation_ids", "test_ids")
            ]
            _, _, test_x, independent_x = standardize_fold(
                list(cohort.features.values()), train, validation, test,
                None if independent is None else list(independent.features.values()),
            )
            model = CrossModalFusionClassifier(
                input_sizes=cohort.input_sizes,
                num_layers=config["effective_num_layers"],
                hidden_size=config["hidden_size"],
                output_size=output_size,
                num_masks=config["num_masks"],
                group_ratio=config["group_ratio"],
                dropout=config["dropout"],
                num_heads=config["num_heads"],
                entmax_alpha=config["entmax_alpha"],
                ablation=AblationConfig(**config["ablation"]),
            )
            model.load_state_dict(torch.load(
                fold_dir / "best_model.pt", map_location="cpu", weights_only=True
            ))
            model.to(device).eval()

            fold_probabilities = infer_conditions(model, test_x, device)
            archived = pd.read_csv(fold_dir / "predictions.csv")
            archived = archived.set_index("ID").loc[cohort.ids[test]]
            columns = probability_columns(archived)
            error = float(np.max(np.abs(
                fold_probabilities[0] - archived[columns].to_numpy()
            )))
            if error >= 1e-5:
                raise AssertionError(f"Archived CV prediction mismatch: {error}")
            cv_probabilities[:, test] = fold_probabilities
            seen[test] += 1

            independent_error = None
            if independent is not None:
                fold_independent = infer_conditions(model, independent_x, device)
                archived_independent = pd.read_csv(
                    fold_dir / "independent_predictions.csv"
                ).set_index("ID").loc[independent.ids]
                columns = probability_columns(archived_independent)
                independent_error = float(np.max(np.abs(
                    fold_independent[0] - archived_independent[columns].to_numpy()
                )))
                if independent_error >= 1e-5:
                    raise AssertionError(
                        f"Archived independent prediction mismatch: {independent_error}"
                    )
                independent_sum += fold_independent / 10.0
            audits.append({
                "cohort": cohort_name,
                "task": task,
                "repeat": repeat,
                "fold": fold,
                "cv_max_error": error,
                "independent_max_error": independent_error,
            })
            del model

        assert np.all(seen == 1) and np.isfinite(cv_probabilities).all()
        evaluations = [("CV", cohort.labels, cv_probabilities)]
        if independent is not None:
            evaluations.append(("Independent", independent.labels, independent_sum))
        for evaluation, labels, probabilities in evaluations:
            if task == "detection":
                targets = [("", labels, probabilities[:, :, 1])]
            else:
                targets = [
                    (cancer, (labels == label).astype(int), probabilities[:, :, label])
                    for label, cancer in enumerate(CANCERS, start=1)
                ]
            for cancer, binary_labels, scores in targets:
                baseline_auc = float(roc_auc_score(binary_labels, scores[0]))
                for index, modality in enumerate(MODALITIES, start=1):
                    occluded_auc = float(roc_auc_score(binary_labels, scores[index]))
                    records.append({
                        "cohort": cohort_name,
                        "task": task,
                        "evaluation": evaluation,
                        "repeat": repeat,
                        "cancer": cancer,
                        "modality": modality,
                        "baseline_auc": baseline_auc,
                        "occluded_auc": occluded_auc,
                        "delta_auc": baseline_auc - occluded_auc,
                    })
    return records, audits


def build_figure(summary):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(14, 8.4))
    outer = figure.add_gridspec(
        2, 1, left=0.065, right=0.95, bottom=0.10, top=0.95,
        hspace=0.50, height_ratios=[1, 1.25]
    )
    top = outer[0].subgridspec(1, 3, wspace=0.40)
    bottom = outer[1].subgridspec(1, 2, wspace=0.30)
    display_names = {
        "CRA001537": "Zhang",
        "PRJNA929650": "Pham",
        "HRA003209": "Bie",
    }
    for panel, cohort_name in enumerate(DETECTION_RUNS):
        axis = figure.add_subplot(top[panel])
        subset = summary[
            (summary.task == "detection") & (summary.cohort == cohort_name)
        ]
        evaluations = ["CV"] + ([] if cohort_name == "CRA001537" else ["Independent"])
        for evaluation_index, evaluation in enumerate(evaluations):
            values = subset[subset.evaluation == evaluation].set_index("modality").loc[MODALITIES]
            offset = 0 if len(evaluations) == 1 else (-0.09 if evaluation_index == 0 else 0.09)
            axis.errorbar(
                np.arange(4) + offset, values["mean"], yerr=values["std"],
                fmt="o" if evaluation_index == 0 else "s", capsize=4,
                color=["#16877A", "#C8752D"][evaluation_index], label=evaluation,
            )
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xticks(range(4), MODALITIES, rotation=15)
        axis.set_ylabel("ROC AUC decrease after modality occlusion")
        axis.set_title(
            f"{'ABC'[panel]}  {display_names[cohort_name]} et al. dataset",
            loc="left", fontweight="bold", fontsize=10,
        )
        axis.legend(frameon=False, fontsize=8, loc="upper right")
        axis.grid(axis="y", alpha=0.18)

    for panel, evaluation in enumerate(("CV", "Independent")):
        axis = figure.add_subplot(bottom[panel])
        subset = summary[(summary.task == "tissue") & (summary.evaluation == evaluation)]
        means = subset.pivot(index="cancer", columns="modality", values="mean").loc[CANCERS, MODALITIES]
        standard_deviations = subset.pivot(
            index="cancer", columns="modality", values="std"
        ).loc[CANCERS, MODALITIES]
        color_limit = max(0.01, float(np.ceil(np.abs(means.to_numpy()).max() * 100) / 100))
        image = axis.imshow(
            means, vmin=-color_limit, vmax=color_limit, cmap="RdBu_r", aspect="auto"
        )
        axis.set_xticks(range(4), MODALITIES, rotation=12)
        axis.set_yticks(range(7), CANCERS)
        axis.set_title(
            f"{'DE'[panel]}  Bie et al. tissue of origin: {evaluation}",
            loc="left", fontweight="bold", fontsize=10,
        )
        for row in range(7):
            for column in range(4):
                label = f"{means.iloc[row, column]:.3f}"
                if evaluation == "CV":
                    label += f"\n+/-{standard_deviations.iloc[row, column]:.3f}"
                axis.text(
                    column, row, label, ha="center", va="center", fontsize=7,
                    color="white" if abs(means.iloc[row, column]) > 0.60 * color_limit else "black",
                )
        figure.colorbar(
            image, ax=axis, fraction=0.035, pad=0.025,
            label="ROC AUC decrease after modality occlusion",
        )

    temporary_pdf = OUT / "revise_3_1.tmp.pdf"
    output_pdf = OUT / "revise_3_1.pdf"
    figure.savefig(temporary_pdf)
    plt.close(figure)
    temporary_pdf.replace(output_pdf)
    shutil.copy2(output_pdf, FINAL)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_records = []
    all_audits = []
    for cohort_name, experiment in DETECTION_RUNS.items():
        records, audits = run_records(cohort_name, "detection", experiment, device)
        all_records.extend(records)
        all_audits.extend(audits)
    records, audits = run_records(
        "HRA003209", "tissue", "bie_occlusion_tissue_full", device
    )
    all_records.extend(records)
    all_audits.extend(audits)

    frame = pd.DataFrame(all_records)
    frame.to_csv(OUT / "modality_occlusion_results.csv", index=False)
    summary = frame.groupby(
        ["cohort", "task", "evaluation", "cancer", "modality"], dropna=False
    ).delta_auc.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(OUT / "modality_occlusion_summary.csv", index=False)
    if not (summary["count"] == 5).all():
        raise AssertionError("Every plotted estimate must contain five paired repeats")
    (OUT / "checkpoint_prediction_audit.json").write_text(
        json.dumps(all_audits, indent=2)
    )
    build_figure(summary)
    print(summary.to_string(index=False), flush=True)
    print("BIE_OCCLUSION_FIGURE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
