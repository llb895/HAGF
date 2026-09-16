"""Strict, ID-aligned loading for the transferred APBC cfDNA matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


MODALITY_ORDER = ("FSR", "EDM", "CNV", "Methylation")
TISSUE_LABELS = {
    "healthy": 0,
    "brca": 1,
    "coread": 2,
    "esca": 3,
    "stad": 4,
    "lihc": 5,
    "nsclc": 6,
    "paca": 7,
}


@dataclass(frozen=True)
class DatasetSpec:
    paths: Mapping[str, str]
    label_index: int
    feature_start: int
    label_mode: str


@dataclass
class LoadedCohort:
    cohort: str
    task: str
    split: str
    ids: np.ndarray
    labels: np.ndarray
    features: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]

    @property
    def modality_names(self) -> list[str]:
        return list(self.features)

    @property
    def input_sizes(self) -> list[int]:
        return [matrix.shape[1] for matrix in self.features.values()]

    def without_modality(self, modality: str | None) -> "LoadedCohort":
        if modality is None:
            return self
        if modality not in self.features:
            raise ValueError(f"unknown modality {modality}; choose from {self.modality_names}")
        return LoadedCohort(
            cohort=self.cohort,
            task=self.task,
            split=self.split,
            ids=self.ids,
            labels=self.labels,
            features={name: value for name, value in self.features.items() if name != modality},
            feature_names={
                name: value for name, value in self.feature_names.items() if name != modality
            },
        )


def _binary_paths(cohort: str, split: str) -> dict[str, str]:
    base = f"data/dataset/{cohort}_{split}"
    if cohort == "CRA001537":
        if split != "cross":
            raise ValueError("CRA001537 has no independent validation split")
        methylation = "methy_selected.csv"
    elif cohort == "PRJNA929650":
        methylation = "methy_new_standard_1.csv"
    elif cohort == "HRA003209":
        methylation = "methy.csv"
    else:
        raise ValueError(f"unsupported cohort: {cohort}")
    return {
        "FSR": f"{base}/FSR.csv",
        "EDM": f"{base}/end_motifs.csv",
        "CNV": f"{base}/CopyNumber.csv",
        "Methylation": f"{base}/{methylation}",
    }


def dataset_spec(cohort: str, task: str, split: str) -> DatasetSpec:
    if task == "detection":
        return DatasetSpec(
            paths=_binary_paths(cohort, split),
            label_index=2 if cohort == "HRA003209" else 1,
            feature_start=4 if cohort == "HRA003209" else 2,
            label_mode="binary_hra" if cohort == "HRA003209" else "numeric",
        )
    if task == "tissue" and cohort == "HRA003209":
        base = f"CRA001537/csv/HRA003209_{split}"
        return DatasetSpec(
            paths={
                "FSR": f"{base}/FSR_adjusted.csv",
                "EDM": f"{base}/end_motifs_adjusted.csv",
                "CNV": f"{base}/CopyNumber_adjusted.csv",
                "Methylation": f"{base}/methy_adjusted.csv",
            },
            label_index=2,
            feature_start=4,
            label_mode="tissue",
        )
    raise ValueError(f"unsupported cohort/task combination: {cohort}/{task}")


def encode_labels(values: pd.Series, mode: str) -> np.ndarray:
    if mode == "numeric":
        labels = pd.to_numeric(values, errors="raise").astype(int).to_numpy()
    else:
        normalized = values.astype(str).str.strip().str.lower()
        if mode == "binary_hra":
            labels = (normalized != "healthy").astype(int).to_numpy()
        elif mode == "tissue":
            unknown = sorted(set(normalized) - set(TISSUE_LABELS))
            if unknown:
                raise ValueError(f"unknown tissue labels: {unknown}")
            labels = normalized.map(TISSUE_LABELS).astype(int).to_numpy()
        else:
            raise ValueError(f"unknown label mode: {mode}")
    return labels


def load_cohort(
    subject_root: str | Path,
    cohort: str,
    task: str = "detection",
    split: str = "cross",
) -> LoadedCohort:
    root = Path(subject_root)
    spec = dataset_spec(cohort, task, split)
    frames: dict[str, pd.DataFrame] = {}
    for modality in MODALITY_ORDER:
        path = root / spec.paths[modality]
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frame.iloc[:, 0].duplicated().any():
            raise ValueError(f"duplicate sample IDs in {path}")
        frames[modality] = frame

    reference = frames[MODALITY_ORDER[0]].iloc[:, 0].astype(str).to_numpy()
    reference_set = set(reference)
    features: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    labels_by_modality: dict[str, np.ndarray] = {}

    for modality, frame in frames.items():
        ids = frame.iloc[:, 0].astype(str)
        if set(ids) != reference_set:
            missing = sorted(reference_set - set(ids))[:10]
            extra = sorted(set(ids) - reference_set)[:10]
            raise ValueError(
                f"sample ID mismatch for {modality}; missing={missing}, extra={extra}"
            )
        aligned = frame.assign(_sample_id=ids).set_index("_sample_id").loc[reference]
        labels_by_modality[modality] = encode_labels(
            aligned.iloc[:, spec.label_index], spec.label_mode
        )
        matrix = aligned.iloc[:, spec.feature_start:].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite feature values in {modality}")
        features[modality] = matrix
        feature_names[modality] = [str(name) for name in aligned.columns[spec.feature_start:]]

    labels = labels_by_modality[MODALITY_ORDER[0]]
    for modality, candidate in labels_by_modality.items():
        if not np.array_equal(labels, candidate):
            raise ValueError(f"label mismatch after ID alignment for {modality}")

    return LoadedCohort(
        cohort=cohort,
        task=task,
        split=split,
        ids=reference,
        labels=labels,
        features=features,
        feature_names=feature_names,
    )
