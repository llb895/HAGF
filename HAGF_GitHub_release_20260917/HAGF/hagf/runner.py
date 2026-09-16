"""Reproducible cross-validation runner for APBC reviewer experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .data import LoadedCohort, load_cohort
from .model import AblationConfig, CrossModalFusionClassifier


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBJECT_ROOT = Path(
    os.environ.get("HAGF_SUBJECT_ROOT", PACKAGE_ROOT / "data")
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("HAGF_OUTPUT_ROOT", PACKAGE_ROOT / "outputs")
)
COHORT_SEEDS = {"CRA001537": 39, "PRJNA929650": 999, "HRA003209": 999}


class MultiModalDataset(Dataset):
    def __init__(self, features: list[np.ndarray], labels: np.ndarray) -> None:
        self.features = [torch.from_numpy(x.astype(np.float32, copy=False)) for x in features]
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return tuple(x[index] for x in self.features) + (self.labels[index],)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ablation_for_variant(name: str) -> tuple[AblationConfig, int | None]:
    settings = {
        "full": AblationConfig(),
        "no_grouping": AblationConfig(use_grouping=False),
        "no_sparse_masks": AblationConfig(use_sparse_masks=False),
        "no_transformer": AblationConfig(use_transformer=False),
        "no_positional_embedding": AblationConfig(use_positional_embedding=False),
        "no_fidelity_path": AblationConfig(use_fidelity_path=False),
        "no_cross_modal_fusion": AblationConfig(use_cross_modal_fusion=False),
    }
    if name == "one_layer":
        return AblationConfig(), 1
    if name not in settings:
        raise ValueError(f"unsupported variant: {name}")
    return settings[name], None


def permute_features(
    cohort: LoadedCohort,
    modality: str | None,
    permutation_seed: int | None,
) -> np.ndarray | None:
    if modality is None:
        return None
    if permutation_seed is None:
        raise ValueError("--permutation-seed is required with --permute-modality")
    if modality not in cohort.features:
        raise ValueError(f"cannot permute missing modality: {modality}")
    rng = np.random.default_rng(permutation_seed)
    order = rng.permutation(cohort.features[modality].shape[1])
    apply_feature_permutation(cohort, modality, order)
    return order


def apply_feature_permutation(
    cohort: LoadedCohort,
    modality: str,
    order: np.ndarray,
) -> None:
    if modality not in cohort.features:
        raise ValueError(f"cannot permute missing modality: {modality}")
    if len(order) != cohort.features[modality].shape[1]:
        raise ValueError(
            f"permutation width mismatch for {modality}: "
            f"expected {cohort.features[modality].shape[1]}, got {len(order)}"
        )
    cohort.features[modality] = cohort.features[modality][:, order]
    cohort.feature_names[modality] = [cohort.feature_names[modality][i] for i in order]


def validate_matching_feature_schema(
    cross_cohort: LoadedCohort,
    independent_cohort: LoadedCohort,
) -> None:
    if cross_cohort.modality_names != independent_cohort.modality_names:
        raise ValueError(
            "cross/independent modality mismatch: "
            f"{cross_cohort.modality_names} != {independent_cohort.modality_names}"
        )
    for modality in cross_cohort.modality_names:
        if cross_cohort.feature_names[modality] != independent_cohort.feature_names[modality]:
            raise ValueError(
                f"cross/independent feature schema mismatch for {modality}"
            )


def standardize_fold(
    matrices: list[np.ndarray],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    independent_matrices: list[np.ndarray] | None = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray] | None,
]:
    if independent_matrices is not None and len(independent_matrices) != len(matrices):
        raise ValueError(
            "cross/independent modality count mismatch during standardization"
        )
    train, val, test = [], [], []
    independent = [] if independent_matrices is not None else None
    for modality_index, matrix in enumerate(matrices):
        scaler = StandardScaler().fit(matrix[train_indices])
        train.append(scaler.transform(matrix[train_indices]).astype(np.float32))
        val.append(scaler.transform(matrix[val_indices]).astype(np.float32))
        test.append(scaler.transform(matrix[test_indices]).astype(np.float32))
        if independent is not None:
            independent.append(
                scaler.transform(independent_matrices[modality_index]).astype(np.float32)
            )
    return train, val, test, independent


def make_loader(
    features: list[np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MultiModalDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def unpack_batch(batch, device: torch.device):
    return [x.to(device, non_blocking=True) for x in batch[:-1]], batch[-1].to(
        device, non_blocking=True
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = unpack_batch(batch, device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            total_count += labels.shape[0]
    return total_loss / max(total_count, 1)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, probabilities_all = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = unpack_batch(batch, device)
            probabilities = torch.softmax(model(inputs), dim=1)
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probabilities_all)


def compute_metrics(task: str, labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if task == "detection":
        return {"auc": float(roc_auc_score(labels, probabilities[:, 1]))}
    predictions = probabilities.argmax(axis=1)
    return {
        "top1_accuracy": float(accuracy_score(labels, predictions)),
        "top2_accuracy": float(
            top_k_accuracy_score(
                labels,
                probabilities,
                k=2,
                labels=np.arange(probabilities.shape[1]),
            )
        ),
    }


def inverse_frequency_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("class-weighted loss requires every class in the training split")
    weights = len(labels) / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def prefixed_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def train_fold(
    model: CrossModalFusionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    class_weights: torch.Tensor | None,
    max_epochs: int,
    patience: int,
) -> tuple[CrossModalFusionClassifier, dict]:
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            inputs, labels = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.shape[0]
            train_count += labels.shape[0]

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def prediction_frame(
    cohort: LoadedCohort,
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": cohort.ids[indices],
            "Label": labels,
            **{
                f"Probability_{class_index}": probabilities[:, class_index]
                for class_index in range(probabilities.shape[1])
            },
        }
    )


def save_fold_outputs(
    fold_dir: Path,
    cohort: LoadedCohort,
    test_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    validation_indices: np.ndarray,
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    independent_cohort: LoadedCohort | None,
    independent_labels: np.ndarray | None,
    independent_probabilities: np.ndarray | None,
    model: CrossModalFusionClassifier,
    metrics: dict,
    training: dict,
    runtime_seconds: float,
    save_checkpoint: bool,
    save_masks: bool,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame(cohort, test_indices, labels, probabilities).to_csv(
        fold_dir / "predictions.csv", index=False
    )
    prediction_frame(
        cohort,
        validation_indices,
        validation_labels,
        validation_probabilities,
    ).to_csv(fold_dir / "validation_predictions.csv", index=False)
    if independent_cohort is not None:
        if independent_labels is None or independent_probabilities is None:
            raise ValueError("independent cohort was provided without predictions")
        prediction_frame(
            independent_cohort,
            np.arange(len(independent_cohort.ids)),
            independent_labels,
            independent_probabilities,
        ).to_csv(fold_dir / "independent_predictions.csv", index=False)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **metrics,
                "best_val_loss": training["best_val_loss"],
                "epochs_ran": training["epochs_ran"],
                "runtime_seconds": runtime_seconds,
            },
            handle,
            indent=2,
        )
    pd.DataFrame(training["history"]).to_csv(fold_dir / "history.csv", index=False)
    if save_checkpoint:
        torch.save(model.state_dict(), fold_dir / "best_model.pt")
    if save_masks:
        arrays = {
            f"branch_{branch}_layer_{layer}": mask.numpy()
            for branch, layer, mask in model.mask_tensors()
        }
        np.savez_compressed(fold_dir / "masks.npz", **arrays)


def save_selection_outputs(
    fold_dir: Path,
    cohort: LoadedCohort,
    validation_indices: np.ndarray,
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_summary: dict,
    training: dict,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame(
        cohort,
        validation_indices,
        validation_labels,
        validation_probabilities,
    ).to_csv(fold_dir / "validation_predictions.csv", index=False)
    with (fold_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_summary, handle, indent=2)
    pd.DataFrame(training["history"]).to_csv(fold_dir / "history.csv", index=False)


def completed_fold_summary(
    fold_dir: Path,
    repeat: int,
    fold: int,
    require_independent: bool = False,
) -> dict | None:
    required = [
        fold_dir / "metrics.json",
        fold_dir / "predictions.csv",
        fold_dir / "history.csv",
        fold_dir / "split.json",
    ]
    if require_independent:
        required.append(fold_dir / "independent_predictions.csv")
    if not all(path.is_file() for path in required):
        return None
    with (fold_dir / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {"repeat": repeat, "fold": fold, **metrics}


def completed_selection_summary(
    fold_dir: Path, repeat: int, fold: int
) -> dict | None:
    required = (
        fold_dir / "validation_metrics.json",
        fold_dir / "validation_predictions.csv",
        fold_dir / "history.csv",
        fold_dir / "split.json",
    )
    if not all(path.is_file() for path in required):
        return None
    with (fold_dir / "validation_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {"repeat": repeat, "fold": fold, **metrics}


def run(args: argparse.Namespace) -> None:
    if args.selection_only and args.evaluate_independent:
        raise ValueError(
            "--selection-only and --evaluate-independent cannot be used together"
        )
    output_root = Path(args.output_root)
    cohort = load_cohort(args.subject_root, args.cohort, args.task, "cross")
    cohort = cohort.without_modality(args.drop_modality)
    original_permuted_feature_names = (
        list(cohort.feature_names[args.permute_modality])
        if args.permute_modality is not None
        else None
    )
    independent_cohort = None
    if args.evaluate_independent:
        independent_cohort = load_cohort(
            args.subject_root, args.cohort, args.task, "validation"
        ).without_modality(args.drop_modality)
        validate_matching_feature_schema(cohort, independent_cohort)
    permutation_order = permute_features(
        cohort, args.permute_modality, args.permutation_seed
    )
    if independent_cohort is not None and permutation_order is not None:
        assert args.permute_modality is not None
        apply_feature_permutation(
            independent_cohort, args.permute_modality, permutation_order
        )

    ablation, forced_layers = ablation_for_variant(args.variant)
    num_layers = forced_layers or args.num_layers
    base_seed = args.seed if args.seed is not None else COHORT_SEEDS[args.cohort]
    experiment_name = args.experiment_name or args.variant
    if args.drop_modality:
        experiment_name += f"_without_{args.drop_modality}"
    if args.permute_modality:
        experiment_name += f"_{args.permute_modality}_perm{args.permutation_seed}"
    run_dir = output_root / args.cohort / args.task / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if permutation_order is not None:
        assert args.permute_modality is not None
        assert original_permuted_feature_names is not None
        np.save(run_dir / "feature_permutation.npy", permutation_order)
        pd.DataFrame(
            {
                "permuted_index": np.arange(len(permutation_order)),
                "original_index": permutation_order,
                "feature_name": [
                    original_permuted_feature_names[index]
                    for index in permutation_order
                ],
            }
        ).to_csv(run_dir / "feature_permutation_mapping.csv", index=False)

    run_config = {
        **vars(args),
        "subject_root": str(args.subject_root),
        "output_root": str(args.output_root),
        "base_seed": base_seed,
        "modalities": cohort.modality_names,
        "input_sizes": cohort.input_sizes,
        "sample_count": len(cohort.ids),
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(cohort.labels, return_counts=True))
        },
        "independent_sample_count": (
            len(independent_cohort.ids) if independent_cohort is not None else None
        ),
        "independent_class_counts": (
            {
                str(label): int(count)
                for label, count in zip(
                    *np.unique(independent_cohort.labels, return_counts=True)
                )
            }
            if independent_cohort is not None
            else None
        ),
        "ablation": asdict(ablation),
        "effective_num_layers": num_layers,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    matrices = list(cohort.features.values())
    independent_matrices = (
        list(independent_cohort.features.values())
        if independent_cohort is not None
        else None
    )
    fold_summaries: list[dict] = []
    folds_completed = 0

    for repeat in range(args.repeats):
        if args.repeat_index is not None and repeat + 1 != args.repeat_index:
            continue
        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=base_seed + repeat,
        )
        for fold_index, (train_val_indices, test_indices) in enumerate(
            splitter.split(cohort.ids, cohort.labels), start=1
        ):
            split_seed = base_seed + repeat * 1000 + fold_index
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=args.validation_fraction,
                random_state=split_seed,
                shuffle=True,
                stratify=cohort.labels[train_val_indices],
            )
            fold_dir = run_dir / f"repeat_{repeat + 1}" / f"fold_{fold_index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            if args.resume:
                if args.selection_only:
                    completed = completed_selection_summary(
                        fold_dir, repeat=repeat + 1, fold=fold_index
                    )
                else:
                    completed = completed_fold_summary(
                        fold_dir,
                        repeat=repeat + 1,
                        fold=fold_index,
                        require_independent=args.evaluate_independent,
                    )
                if completed is not None:
                    fold_summaries.append(completed)
                    pd.DataFrame(fold_summaries).sort_values(
                        ["repeat", "fold"]
                    ).to_csv(
                        run_dir
                        / (
                            "fold_validation_metrics.csv"
                            if args.selection_only
                            else "fold_metrics.csv"
                        ),
                        index=False,
                    )
                    print(
                        json.dumps(
                            {
                                "repeat": repeat + 1,
                                "fold": fold_index,
                                "status": "skipped_completed",
                            }
                        ),
                        flush=True,
                    )
                    folds_completed += 1
                    if args.max_folds and folds_completed >= args.max_folds:
                        print(f"RUN_COMPLETE {run_dir}", flush=True)
                        return
                    continue
            with (fold_dir / "split.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "train_ids": cohort.ids[train_indices].tolist(),
                        "validation_ids": cohort.ids[val_indices].tolist(),
                        "test_ids": cohort.ids[test_indices].tolist(),
                    },
                    handle,
                    indent=2,
                )

            train_x, val_x, test_x, independent_x = standardize_fold(
                matrices,
                train_indices,
                val_indices,
                test_indices,
                independent_matrices,
            )
            train_loader = make_loader(
                train_x,
                cohort.labels[train_indices],
                args.batch_size,
                True,
                split_seed,
            )
            val_loader = make_loader(
                val_x,
                cohort.labels[val_indices],
                args.eval_batch_size,
                False,
                split_seed,
            )
            test_loader = None
            independent_loader = None
            if not args.selection_only:
                test_loader = make_loader(
                    test_x,
                    cohort.labels[test_indices],
                    args.eval_batch_size,
                    False,
                    split_seed,
                )
                if independent_cohort is not None:
                    assert independent_x is not None
                    independent_loader = make_loader(
                        independent_x,
                        independent_cohort.labels,
                        args.eval_batch_size,
                        False,
                        split_seed,
                    )

            seed_everything(split_seed)
            model = CrossModalFusionClassifier(
                input_sizes=cohort.input_sizes,
                num_layers=num_layers,
                hidden_size=args.hidden_size,
                output_size=2 if args.task == "detection" else 8,
                num_masks=args.num_masks,
                group_ratio=args.group_ratio,
                dropout=args.dropout,
                num_heads=args.num_heads,
                entmax_alpha=args.entmax_alpha,
                ablation=ablation,
            ).to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            class_weights = None
            if args.class_weighted_loss:
                class_weights = inverse_frequency_class_weights(
                    cohort.labels[train_indices],
                    2 if args.task == "detection" else 8,
                    device,
                )

            started = time.perf_counter()
            model, training = train_fold(
                model,
                train_loader,
                val_loader,
                device,
                args.learning_rate,
                args.weight_decay,
                class_weights,
                args.max_epochs,
                args.patience,
            )
            validation_labels, validation_probabilities = predict(
                model, val_loader, device
            )
            validation_metrics = prefixed_metrics(
                compute_metrics(args.task, validation_labels, validation_probabilities),
                "validation",
            )
            runtime_seconds = time.perf_counter() - started

            if args.selection_only:
                summary = {
                    "repeat": repeat + 1,
                    "fold": fold_index,
                    **validation_metrics,
                    "parameter_count": parameter_count,
                    "best_val_loss": training["best_val_loss"],
                    "epochs_ran": training["epochs_ran"],
                    "runtime_seconds": runtime_seconds,
                }
                save_selection_outputs(
                    fold_dir,
                    cohort,
                    val_indices,
                    validation_labels,
                    validation_probabilities,
                    {key: value for key, value in summary.items() if key not in ("repeat", "fold")},
                    training,
                )
                fold_summaries.append(summary)
                pd.DataFrame(fold_summaries).sort_values(["repeat", "fold"]).to_csv(
                    run_dir / "fold_validation_metrics.csv", index=False
                )
                print(json.dumps(summary), flush=True)
                folds_completed += 1
                if args.max_folds and folds_completed >= args.max_folds:
                    print(f"RUN_COMPLETE {run_dir}", flush=True)
                    return
                continue

            assert test_loader is not None
            labels, probabilities = predict(model, test_loader, device)
            metrics = compute_metrics(args.task, labels, probabilities)
            metrics["parameter_count"] = parameter_count
            metrics.update(validation_metrics)
            independent_labels = None
            independent_probabilities = None
            if independent_loader is not None:
                independent_labels, independent_probabilities = predict(
                    model, independent_loader, device
                )
                metrics.update(
                    prefixed_metrics(
                        compute_metrics(
                            args.task,
                            independent_labels,
                            independent_probabilities,
                        ),
                        "independent",
                    )
                )
            runtime_seconds = time.perf_counter() - started
            save_fold_outputs(
                fold_dir,
                cohort,
                test_indices,
                labels,
                probabilities,
                val_indices,
                validation_labels,
                validation_probabilities,
                independent_cohort,
                independent_labels,
                independent_probabilities,
                model,
                metrics,
                training,
                runtime_seconds,
                args.save_checkpoints,
                args.save_masks,
            )
            summary = {
                "repeat": repeat + 1,
                "fold": fold_index,
                **metrics,
                "best_val_loss": training["best_val_loss"],
                "epochs_ran": training["epochs_ran"],
                "runtime_seconds": runtime_seconds,
            }
            fold_summaries.append(summary)
            pd.DataFrame(fold_summaries).sort_values(["repeat", "fold"]).to_csv(
                run_dir / "fold_metrics.csv", index=False
            )
            print(json.dumps(summary), flush=True)

            folds_completed += 1
            if args.max_folds and folds_completed >= args.max_folds:
                print(f"RUN_COMPLETE {run_dir}", flush=True)
                return

    print(f"RUN_COMPLETE {run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, default=DEFAULT_SUBJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cohort", required=True, choices=COHORT_SEEDS)
    parser.add_argument("--task", default="detection", choices=("detection", "tissue"))
    parser.add_argument(
        "--variant",
        default="full",
        choices=(
            "full",
            "no_grouping",
            "no_sparse_masks",
            "no_transformer",
            "no_positional_embedding",
            "no_fidelity_path",
            "no_cross_modal_fusion",
            "one_layer",
        ),
    )
    parser.add_argument("--drop-modality", choices=("FSR", "EDM", "CNV", "Methylation"))
    parser.add_argument("--permute-modality", choices=("FSR", "EDM", "CNV", "Methylation"))
    parser.add_argument("--permutation-seed", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--group-ratio", type=float, default=0.2)
    parser.add_argument("--num-masks", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--entmax-alpha", type=float, default=1.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--class-weighted-loss", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--repeat-index", type=int, choices=range(1, 6))
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--evaluate-independent", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
