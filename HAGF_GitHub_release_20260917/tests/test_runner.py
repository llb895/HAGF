import numpy as np
import torch

from hagf.data import LoadedCohort
from hagf.runner import (
    apply_feature_permutation,
    inverse_frequency_class_weights,
    prefixed_metrics,
    standardize_fold,
    validate_matching_feature_schema,
)


def test_inverse_frequency_class_weights() -> None:
    weights = inverse_frequency_class_weights(
        np.asarray([0, 0, 0, 1]), num_classes=2, device=torch.device("cpu")
    )
    assert torch.allclose(weights, torch.tensor([2.0 / 3.0, 2.0]))


def test_inverse_frequency_class_weights_rejects_missing_class() -> None:
    try:
        inverse_frequency_class_weights(
            np.asarray([0, 0]), num_classes=2, device=torch.device("cpu")
        )
    except ValueError as error:
        assert "every class" in str(error)
    else:
        raise AssertionError("Expected a ValueError for a missing training class")


def test_prefixed_metrics() -> None:
    assert prefixed_metrics({"auc": 0.75}, "validation") == {
        "validation_auc": 0.75
    }


def test_standardize_fold_uses_training_samples_only() -> None:
    matrix = np.asarray([[0.0], [2.0], [20.0], [30.0]], dtype=np.float32)
    independent_matrix = np.asarray([[100.0]], dtype=np.float32)
    train, validation, test, independent = standardize_fold(
        [matrix],
        np.asarray([0, 1]),
        np.asarray([2]),
        np.asarray([3]),
        [independent_matrix],
    )
    assert np.allclose(train[0].ravel(), [-1.0, 1.0])
    assert np.allclose(validation[0].ravel(), [19.0])
    assert np.allclose(test[0].ravel(), [29.0])
    assert independent is not None
    assert np.allclose(independent[0].ravel(), [99.0])


def make_cohort(split: str, rows: int = 2) -> LoadedCohort:
    return LoadedCohort(
        cohort="PRJNA929650",
        task="detection",
        split=split,
        ids=np.asarray([f"sample_{index}" for index in range(rows)]),
        labels=np.asarray([index % 2 for index in range(rows)]),
        features={
            "FSR": np.asarray(
                [[row * 10 + column for column in range(3)] for row in range(rows)],
                dtype=np.float32,
            )
        },
        feature_names={"FSR": ["a", "b", "c"]},
    )


def test_feature_permutation_can_be_reused_for_independent_data() -> None:
    cohort = make_cohort("cross")
    independent = make_cohort("validation")
    order = np.asarray([2, 0, 1])
    apply_feature_permutation(cohort, "FSR", order)
    apply_feature_permutation(independent, "FSR", order)
    assert cohort.feature_names["FSR"] == ["c", "a", "b"]
    assert independent.feature_names["FSR"] == ["c", "a", "b"]
    assert np.array_equal(cohort.features["FSR"][0], [2.0, 0.0, 1.0])


def test_feature_schema_validation_rejects_different_order() -> None:
    cohort = make_cohort("cross")
    independent = make_cohort("validation")
    independent.feature_names["FSR"] = ["b", "a", "c"]
    try:
        validate_matching_feature_schema(cohort, independent)
    except ValueError as error:
        assert "feature schema mismatch" in str(error)
    else:
        raise AssertionError("Expected a ValueError for mismatched feature order")


if __name__ == "__main__":
    test_inverse_frequency_class_weights()
    test_inverse_frequency_class_weights_rejects_missing_class()
    test_prefixed_metrics()
    test_standardize_fold_uses_training_samples_only()
    test_feature_permutation_can_be_reused_for_independent_data()
    test_feature_schema_validation_rejects_different_order()
    print("ALL_RUNNER_TESTS_PASSED")
