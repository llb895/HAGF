import os
from pathlib import Path

import numpy as np

from hagf.data import load_cohort


ROOT = Path(
    os.environ.get(
        "HAGF_SUBJECT_ROOT",
        Path(__file__).resolve().parents[1] / "data",
    )
)


def check(cohort_name: str, task: str, split: str, expected_samples: int) -> None:
    cohort = load_cohort(ROOT, cohort_name, task, split)
    assert len(cohort.ids) == expected_samples
    assert len(set(cohort.ids)) == expected_samples
    assert cohort.modality_names == ["FSR", "EDM", "CNV", "Methylation"]
    assert all(len(matrix) == expected_samples for matrix in cohort.features.values())
    assert all(np.isfinite(matrix).all() for matrix in cohort.features.values())


if __name__ == "__main__":
    check("CRA001537", "detection", "cross", 60)
    check("PRJNA929650", "detection", "cross", 347)
    check("PRJNA929650", "detection", "validation", 169)
    check("HRA003209", "detection", "cross", 894)
    check("HRA003209", "detection", "validation", 383)
    check("HRA003209", "tissue", "cross", 894)
    check("HRA003209", "tissue", "validation", 383)
    print("ALL_DATA_TESTS_PASSED")
