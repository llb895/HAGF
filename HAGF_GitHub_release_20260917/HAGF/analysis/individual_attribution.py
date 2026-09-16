"""Frozen-checkpoint individual attribution analysis for APBC reviewer 2 comment 5.

The script performs no fitting or model selection. It reuses the ten fold models
from repeat 1 of the Pham et al. independent-validation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hagf.data import load_cohort
from hagf.model import AblationConfig, CrossModalFusionClassifier
from hagf.runner import standardize_fold


ROOT = Path(
    os.environ.get("HAGF_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)
RUN = ROOT / "revision_workspace/runs/PRJNA929650/detection/full_independent"
OUT = ROOT / "revision_workspace/response_evidence/individual_attribution"
MODALITIES = ["FSR", "EDM", "CNV", "Methylation"]
COLORS = {
    "FSR": "#4477AA",
    "EDM": "#CC6677",
    "CNV": "#228833",
    "Methylation": "#EEAA33",
}


def build_model(cfg: dict) -> CrossModalFusionClassifier:
    return CrossModalFusionClassifier(
        input_sizes=cfg["input_sizes"],
        num_layers=cfg["effective_num_layers"],
        hidden_size=cfg["hidden_size"],
        output_size=2,
        num_masks=cfg["num_masks"],
        group_ratio=cfg["group_ratio"],
        dropout=cfg["dropout"],
        num_heads=cfg["num_heads"],
        entmax_alpha=cfg["entmax_alpha"],
        ablation=AblationConfig(**cfg["ablation"]),
    )


def fold_inputs(cfg: dict, cohort, independent, fold_dir: Path):
    lookup = {str(sample_id): index for index, sample_id in enumerate(cohort.ids)}
    split = json.loads((fold_dir / "split.json").read_text())
    train, validation, test = [
        np.asarray([lookup[str(sample_id)] for sample_id in split[key]], dtype=int)
        for key in ["train_ids", "validation_ids", "test_ids"]
    ]
    _, _, _, independent_x = standardize_fold(
        list(cohort.features.values()),
        train,
        validation,
        test,
        list(independent.features.values()),
    )
    return independent_x


def predict(model, inputs, device, batch_size=128):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs[0]), batch_size):
            batch = [
                torch.as_tensor(x[start : start + batch_size], device=device)
                for x in inputs
            ]
            outputs.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outputs)


def integrated_gradients(model, inputs, device, steps=32, batch_size=12):
    """Integrated gradients of cancer probability from a zero baseline."""
    model.eval()
    attr_parts = [[] for _ in inputs]
    completeness = []
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(steps)
    alphas = torch.as_tensor((nodes + 1.0) / 2.0, device=device, dtype=torch.float32)
    weights = torch.as_tensor(quadrature_weights / 2.0, device=device, dtype=torch.float32)
    for start in range(0, len(inputs[0]), batch_size):
        original = [
            torch.as_tensor(x[start : start + batch_size], device=device)
            for x in inputs
        ]
        batch_count = original[0].shape[0]
        scaled = [
            torch.cat([x * alpha for alpha in alphas], dim=0).detach().requires_grad_(True)
            for x in original
        ]
        probability = torch.softmax(model(scaled), dim=1)[:, 1]
        gradients = torch.autograd.grad(probability.sum(), scaled)
        attributes = []
        for x, gradient in zip(original, gradients):
            gradient = gradient.reshape(steps, batch_count, -1)
            integrated = (gradient * weights[:, None, None]).sum(dim=0)
            attributes.append(x * integrated)
        zeros = [torch.zeros_like(x) for x in original]
        with torch.inference_mode():
            observed = torch.softmax(model(original), dim=1)[:, 1]
            baseline = torch.softmax(model(zeros), dim=1)[:, 1]
            probability_difference = observed - baseline
            attributed = sum(value.sum(dim=1) for value in attributes)
            completeness.extend((attributed - probability_difference).abs().cpu().numpy())
        for modality_index, value in enumerate(attributes):
            attr_parts[modality_index].append(value.cpu().numpy())
    return [np.concatenate(values) for values in attr_parts], np.asarray(completeness)


def first_layer_gate_scores(model):
    """Map first-layer masks to original features, relative to a uniform mask."""
    scores = []
    for branch in model.branches:
        layer = branch.layers[0]
        groups = layer.normalized_masks()
        parts = []
        for group in groups:
            width = group.shape[1]
            parts.append(group.detach().cpu().numpy().mean(axis=0) * width)
        scores.append(np.concatenate(parts))
    return scores


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = max(1, min(count, values.shape[1]))
    return np.argpartition(values, -count, axis=1)[:, -count:]


def zero_selected(inputs, selected, offsets):
    changed = [x.copy() for x in inputs]
    rows = np.arange(selected.shape[0])[:, None]
    for modality_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        local = selected - start
        valid = (local >= 0) & (local < end - start)
        if valid.any():
            row_index, col_index = np.where(valid)
            changed[modality_index][row_index, local[row_index, col_index]] = 0.0
    return changed


def format_feature(modality: str, name: str) -> str:
    if modality in {"CNV", "Methylation"}:
        parts = str(name).split("-")
        if len(parts) >= 3:
            start = int(parts[1]) / 1_000_000
            end = int(parts[2]) / 1_000_000
            return f"{modality} chr{parts[0]}:{start:.1f}-{end:.1f} Mb"
    if modality == "FSR":
        return f"FSR bin {name}"
    return f"EDM {name}"


def save_tiff(fig, path: Path):
    fig.savefig(
        path,
        dpi=300,
        format="tiff",
        pil_kwargs={"compression": "raw"},
        facecolor="white",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--attribution-per-class", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RUN / "run_config.json").read_text())
    cohort = load_cohort(Path(cfg["subject_root"]), "PRJNA929650", "detection", "cross")
    independent = load_cohort(
        Path(cfg["subject_root"]), "PRJNA929650", "detection", "validation"
    )
    assert cohort.feature_names == independent.feature_names
    selection_probability = (
        pd.read_csv(RUN / "repeat_1/fold_1/independent_predictions.csv")
        .set_index("ID")
        .loc[independent.ids]
        .Probability_1.to_numpy()
    )
    analysis_indices = []
    for target in [0, 1]:
        class_indices = np.where(independent.labels == target)[0]
        ordered = class_indices[np.argsort(selection_probability[class_indices])]
        count = min(args.attribution_per_class, len(ordered))
        positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        analysis_indices.extend(ordered[positions].tolist())
    analysis_indices = np.asarray(analysis_indices, dtype=int)
    analysis_ids = independent.ids[analysis_indices]
    analysis_labels = independent.labels[analysis_indices]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    folds = [1] if args.smoke else list(range(1, args.folds + 1))
    attr_sum = [
        np.zeros((len(analysis_indices), x.shape[1]), dtype=np.float64)
        for x in independent.features.values()
    ]
    gate_sum = [np.zeros(x.shape[1], dtype=np.float64) for x in independent.features.values()]
    probability_sum = np.zeros(len(analysis_indices), dtype=np.float64)
    audits = []

    for fold in folds:
        fold_dir = RUN / f"repeat_1/fold_{fold}"
        cache = OUT / f"repeat_1_fold_{fold}_ig{args.steps}.npz"
        model = build_model(cfg)
        checkpoint = fold_dir / "best_model.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.to(device).eval()
        inputs = fold_inputs(cfg, cohort, independent, fold_dir)
        probability_full = predict(model, inputs, device)
        archived = (
            pd.read_csv(fold_dir / "independent_predictions.csv")
            .set_index("ID")
            .loc[independent.ids]
        )
        max_error = float(
            np.max(np.abs(probability_full - archived.Probability_1.to_numpy()))
        )
        assert max_error < 1e-5, (fold, max_error)
        probability = probability_full[analysis_indices]
        analysis_inputs = [x[analysis_indices] for x in inputs]
        if cache.exists() and not args.smoke:
            saved = np.load(cache)
            attributes = [saved[f"attr_{name}"] for name in MODALITIES]
            completeness = saved["completeness"]
        else:
            subset = [x[:2] for x in inputs] if args.smoke else analysis_inputs
            attributes, completeness = integrated_gradients(
                model, subset, device, steps=args.steps, batch_size=4 if args.smoke else 12
            )
            if not args.smoke:
                np.savez_compressed(
                    cache,
                    **{f"attr_{name}": value for name, value in zip(MODALITIES, attributes)},
                    completeness=completeness,
                )
        gate_scores = first_layer_gate_scores(model)
        if args.smoke:
            smoke_summary = {
                "fold": fold,
                "prediction_max_error": max_error,
                "ig_mean_abs_completeness_error": float(completeness.mean()),
                "ig_max_abs_completeness_error": float(completeness.max()),
            }
            (OUT / "smoke_summary.json").write_text(json.dumps(smoke_summary, indent=2))
            print(json.dumps(smoke_summary, indent=2))
            return
        for index in range(4):
            attr_sum[index] += attributes[index]
            gate_sum[index] += gate_scores[index]
        probability_sum += probability
        audits.append(
            {
                "repeat": 1,
                "fold": fold,
                "prediction_max_error": max_error,
                "ig_mean_abs_completeness_error": float(completeness.mean()),
                "ig_max_abs_completeness_error": float(completeness.max()),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
        del model
        torch.cuda.empty_cache()
        print(f"Attribution fold {fold}/{len(folds)} complete", flush=True)

    divisor = len(folds)
    attributes = [value / divisor for value in attr_sum]
    gate_scores = [value / divisor for value in gate_sum]
    probabilities = probability_sum / divisor
    flat_attr = np.concatenate(attributes, axis=1)
    flat_gate = np.concatenate(gate_scores)
    sizes = [x.shape[1] for x in independent.features.values()]
    offsets = np.cumsum([0] + sizes)
    labels = analysis_labels

    feature_records = []
    correlations = {}
    for modality, names, gate, attr in zip(
        MODALITIES, independent.feature_names.values(), gate_scores, attributes
    ):
        mean_abs = np.mean(np.abs(attr), axis=0)
        rho, p_value = spearmanr(gate, mean_abs)
        correlations[modality] = {"rho": float(rho), "p": float(p_value)}
        for name, gate_value, attr_value in zip(names, gate, mean_abs):
            feature_records.append(
                {
                    "modality": modality,
                    "feature": name,
                    "gate_weight_relative_to_uniform": float(gate_value),
                    "mean_absolute_ig": float(attr_value),
                }
            )
    pd.DataFrame(feature_records).to_csv(OUT / "feature_attribution.csv", index=False)

    modality_abs = np.column_stack([np.abs(x).sum(axis=1) for x in attributes])
    modality_share = modality_abs / np.maximum(modality_abs.sum(axis=1, keepdims=True), 1e-12)
    pd.DataFrame(
        modality_share, columns=MODALITIES, index=analysis_ids
    ).assign(label=labels, probability=probabilities).to_csv(OUT / "sample_attribution.csv")

    correct_cancer = np.where((labels == 1) & (probabilities >= 0.5))[0]
    correct_healthy = np.where((labels == 0) & (probabilities < 0.5))[0]
    assert len(correct_cancer) and len(correct_healthy)

    def median_representative(indices):
        median = np.median(probabilities[indices])
        return int(indices[np.argmin(np.abs(probabilities[indices] - median))])

    representatives = {
        "cancer": median_representative(correct_cancer),
        "healthy": median_representative(correct_healthy),
    }

    ratios = [0.0, 0.01, 0.05, 0.10, 0.20]
    cancer_attr = flat_attr[correct_cancer]
    cancer_gate = np.broadcast_to(flat_gate, cancer_attr.shape)
    rng = np.random.default_rng(20260912)
    random_repeats = 10
    deletion_predictions = {
        "IG": np.zeros((len(ratios), len(correct_cancer))),
        "Gate": np.zeros((len(ratios), len(correct_cancer))),
        "Random": np.zeros((random_repeats, len(ratios), len(correct_cancer))),
    }

    for fold in folds:
        fold_dir = RUN / f"repeat_1/fold_{fold}"
        model = build_model(cfg)
        model.load_state_dict(
            torch.load(fold_dir / "best_model.pt", map_location="cpu", weights_only=True)
        )
        model.to(device).eval()
        fold_all = fold_inputs(cfg, cohort, independent, fold_dir)
        fold_analysis = [x[analysis_indices] for x in fold_all]
        fold_cancer = [x[correct_cancer] for x in fold_analysis]
        base = predict(model, fold_cancer, device)
        deletion_predictions["IG"][0] += base / divisor
        deletion_predictions["Gate"][0] += base / divisor
        deletion_predictions["Random"][:, 0] += base[None, :] / divisor
        for ratio_index, ratio in enumerate(ratios[1:], start=1):
            count = int(np.ceil(ratio * flat_attr.shape[1]))
            ig_selected = top_indices(cancer_attr, count)
            gate_selected = top_indices(cancer_gate, count)
            deletion_predictions["IG"][ratio_index] += predict(
                model, zero_selected(fold_cancer, ig_selected, offsets), device
            ) / divisor
            deletion_predictions["Gate"][ratio_index] += predict(
                model, zero_selected(fold_cancer, gate_selected, offsets), device
            ) / divisor
            for random_index in range(random_repeats):
                random_selected = np.vstack(
                    [rng.choice(flat_attr.shape[1], count, replace=False) for _ in correct_cancer]
                )
                deletion_predictions["Random"][random_index, ratio_index] += predict(
                    model, zero_selected(fold_cancer, random_selected, offsets), device
                ) / divisor
        del model
        torch.cuda.empty_cache()
        print(f"Deletion fold {fold}/{len(folds)} complete", flush=True)

    base_mean = deletion_predictions["IG"][0].mean()
    deletion_rows = []
    for strategy in ["IG", "Gate"]:
        for ratio, values in zip(ratios, deletion_predictions[strategy]):
            deletion_rows.append(
                {
                    "strategy": strategy,
                    "fraction_removed": ratio,
                    "mean_probability": float(values.mean()),
                    "mean_probability_decrease": float(base_mean - values.mean()),
                    "sd_across_samples": float(values.std(ddof=1)),
                }
            )
    for random_index, matrix in enumerate(deletion_predictions["Random"]):
        for ratio, values in zip(ratios, matrix):
            deletion_rows.append(
                {
                    "strategy": "Random",
                    "random_repeat": random_index,
                    "fraction_removed": ratio,
                    "mean_probability": float(values.mean()),
                    "mean_probability_decrease": float(base_mean - values.mean()),
                    "sd_across_samples": float(values.std(ddof=1)),
                }
            )
    deletion_df = pd.DataFrame(deletion_rows)
    deletion_df.to_csv(OUT / "deletion_fidelity.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(14.0, 10.5), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.08, 1.0, 1.0], hspace=0.66, wspace=0.46)

    ax = fig.add_subplot(grid[0, 0])
    start = 0
    for modality, size in zip(MODALITIES, sizes):
        values = np.mean(np.abs(flat_attr[:, start : start + size]), axis=0)
        gate = flat_gate[start : start + size]
        ax.scatter(
            gate + 1e-4,
            values + 1e-8,
            s=16,
            alpha=0.45,
            color=COLORS[modality],
            edgecolors="none",
            label=f"{modality} (rho={correlations[modality]['rho']:.2f})",
        )
        start += size
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gate weight / uniform weight")
    ax.set_ylabel("Mean absolute IG")
    ax.set_title("A  Global gates vs local attributions", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8, fontsize=12)
    ax.grid(alpha=0.18)

    ax = fig.add_subplot(grid[0, 1])
    selected = []
    for class_indices in [np.where(labels == 0)[0], np.where(labels == 1)[0]]:
        ordered = class_indices[np.argsort(probabilities[class_indices])]
        positions = np.linspace(0, len(ordered) - 1, 15).round().astype(int)
        selected.extend(ordered[positions].tolist())
    selected = np.asarray(selected)
    order = selected[np.argsort(probabilities[selected])]
    bottom = np.zeros(len(order))
    for modality_index, modality in enumerate(MODALITIES):
        values = modality_share[order, modality_index]
        ax.bar(np.arange(len(order)), values, bottom=bottom, width=0.86, color=COLORS[modality], label=modality)
        bottom += values
    ax.axvline(14.5, color="#666666", lw=0.9, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xticks([7, 22], ["Healthy", "Cancer"])
    ax.set_ylabel("Absolute attribution share")
    ax.set_title("B  Individual attribution profiles", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.55),
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.18)

    feature_modalities = np.concatenate(
        [np.repeat(modality, size) for modality, size in zip(MODALITIES, sizes)]
    )
    feature_names = np.concatenate([np.asarray(names) for names in independent.feature_names.values()])
    for panel_index, (kind, panel) in enumerate([("cancer", "C"), ("healthy", "D")]):
        ax = fig.add_subplot(grid[1, panel_index])
        sample_index = representatives[kind]
        support = flat_attr[sample_index] if kind == "cancer" else -flat_attr[sample_index]
        top_by_modality = []
        start = 0
        for size in sizes:
            local = np.argsort(support[start : start + size])[-2:] + start
            top_by_modality.extend(local.tolist())
            start += size
        top = np.asarray(top_by_modality, dtype=int)
        top = top[np.argsort(support[top])]
        labels_text = [format_feature(feature_modalities[i], feature_names[i]) for i in top]
        colors = [COLORS[feature_modalities[i]] for i in top]
        ax.barh(np.arange(len(top)), support[top], color=colors)
        ax.set_yticks(np.arange(len(top)), labels_text)
        ax.set_xlabel(f"IG support for predicted {kind} class")
        ax.set_title(
            f"{panel}  Representative {kind} prediction\nP(cancer) = {probabilities[sample_index]:.3f}",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.18)

    ax = fig.add_subplot(grid[2, :])
    for strategy, color, marker in [("IG", "#7A3E9D", "o"), ("Gate", "#16877A", "s")]:
        sub = deletion_df[deletion_df.strategy == strategy].drop_duplicates("fraction_removed")
        ax.plot(
            100 * sub.fraction_removed,
            sub.mean_probability_decrease,
            marker=marker,
            lw=2,
            color=color,
            label=f"Top {strategy} features",
        )
    random_summary = (
        deletion_df[deletion_df.strategy == "Random"]
        .groupby(["random_repeat", "fraction_removed"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby("fraction_removed")
        .mean(numeric_only=True)
    )
    random_sd = (
        deletion_df[deletion_df.strategy == "Random"]
        .groupby(["random_repeat", "fraction_removed"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby("fraction_removed")["mean_probability_decrease"]
        .std(ddof=1)
    )
    x = 100 * random_summary.index.to_numpy()
    y = random_summary.mean_probability_decrease.to_numpy()
    sd = random_sd.fillna(0).to_numpy()
    ax.plot(x, y, marker="^", lw=1.8, color="#777777", label="Matched random features")
    ax.fill_between(x, y - sd, y + sd, color="#AAAAAA", alpha=0.25, linewidth=0)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Features replaced by the fold-training mean (%)")
    ax.set_ylabel("Mean probability decrease")
    ax.set_title(
        "E  Deletion-based prediction fidelity",
        loc="left",
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center right",
        bbox_to_anchor=(0.985, 0.48),
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.18)

    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.07, top=0.98)
    pdf = OUT / "revise_2_5.pdf"
    tiff = OUT / "Supplementary_Figure_S13.tiff"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    save_tiff(fig, tiff)
    plt.close(fig)

    summary = {
        "cohort": "PRJNA929650 (Pham et al.)",
        "evaluation": "independent validation",
        "independent_cohort_samples": int(len(independent.labels)),
        "attribution_samples": int(len(labels)),
        "attribution_sampling": (
            f"{args.attribution_per_class} samples per class selected at evenly spaced "
            "quantiles of the prespecified fold-1 prediction score"
        ),
        "attribution_cancer_samples": int((labels == 1).sum()),
        "attribution_healthy_samples": int((labels == 0).sum()),
        "correctly_classified_cancer_samples_in_deletion_test": int(len(correct_cancer)),
        "models": int(len(folds)),
        "integrated_gradients_steps": int(args.steps),
        "baseline_zero_interpretation": "fold-training mean after standardization",
        "correlations": correlations,
        "representatives": {
            key: {
                "sample_id": str(analysis_ids[index]),
                "label": int(labels[index]),
                "cancer_probability": float(probabilities[index]),
                "selection_rule": "correctly classified sample closest to the within-class median cancer probability",
            }
            for key, index in representatives.items()
        },
        "deletion_mean_probability_decrease_at_10_percent": {
            strategy: float(
                deletion_df[
                    (deletion_df.strategy == strategy)
                    & np.isclose(deletion_df.fraction_removed, 0.10)
                ].mean_probability_decrease.mean()
            )
            for strategy in ["IG", "Gate", "Random"]
        },
        "checkpoint_audits": audits,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
