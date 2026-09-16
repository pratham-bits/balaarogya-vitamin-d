from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.multimodal_ablations import (
    ABLATION_FEATURE_GROUPS,
    PROFILE_GROWTH_COLUMNS,
    BEHAVIOR_NUTRITION_COLUMNS,
    OPTICAL_ONLY_COLUMNS,
    ablation_experiment_names,
    get_ablation_columns,
)
from src.features.multimodal_columns import multimodal_feature_names
from src.models.multimodal_ablation import (
    build_ablation_model,
    evaluate_ablation,
    cross_validate_ablation,
)


def make_sample_dataframe(
    n_samples: int = 20,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create deterministic synthetic data for testing the ablation
    pipeline.

    This data is ONLY for software/interface testing.
    It must not be used as evidence of model performance.
    """

    rng = np.random.default_rng(42)

    data: dict[str, list] = {}

    numeric_columns = [
        column
        for column in multimodal_feature_names()
        if (
            column == "profile__age_months"
            or column.startswith("growth__")
            or column == "breastfeeding__duration_months"
            or column.startswith("optical__")
        )
    ]

    categorical_columns = [
        column
        for column in multimodal_feature_names()
        if column not in numeric_columns
    ]

    for column in numeric_columns:
        data[column] = rng.normal(
            loc=0.0,
            scale=1.0,
            size=n_samples,
        ).tolist()

    for column in categorical_columns:
        data[column] = [
            "category_a" if i % 2 == 0 else "category_b"
            for i in range(n_samples)
        ]

    X = pd.DataFrame(data)

    y = (
        50.0
        + 5.0 * X["profile__age_months"].to_numpy()
        + rng.normal(0.0, 1.0, n_samples)
    )

    return X, y


def test_expected_ablation_experiments_exist():
    expected = {
        "profile_growth",
        "behavior_nutrition",
        "optical_only",
        "profile_behavior",
        "profile_optical",
        "behavior_optical",
        "all_modalities",
    }

    assert set(ablation_experiment_names()) == expected


def test_profile_growth_group_is_correct():
    assert (
        get_ablation_columns("profile_growth")
        == PROFILE_GROWTH_COLUMNS
    )


def test_behavior_nutrition_group_is_correct():
    assert (
        get_ablation_columns("behavior_nutrition")
        == BEHAVIOR_NUTRITION_COLUMNS
    )


def test_optical_group_contains_exactly_18_features():
    columns = get_ablation_columns("optical_only")

    assert columns == OPTICAL_ONLY_COLUMNS
    assert len(columns) == 18
    assert all(
        column.startswith("optical__")
        for column in columns
    )


def test_ablation_groups_have_no_duplicate_columns():
    for name, columns in ABLATION_FEATURE_GROUPS.items():
        assert len(columns) == len(set(columns)), name


def test_all_modalities_matches_complete_contract():
    assert set(
        get_ablation_columns("all_modalities")
    ) == set(multimodal_feature_names())


def test_combined_ablation_groups_are_correct():
    assert set(
        get_ablation_columns("profile_behavior")
    ) == set(
        (*PROFILE_GROWTH_COLUMNS, *BEHAVIOR_NUTRITION_COLUMNS)
    )

    assert set(
        get_ablation_columns("profile_optical")
    ) == set(
        (*PROFILE_GROWTH_COLUMNS, *OPTICAL_ONLY_COLUMNS)
    )

    assert set(
        get_ablation_columns("behavior_optical")
    ) == set(
        (*BEHAVIOR_NUTRITION_COLUMNS, *OPTICAL_ONLY_COLUMNS)
    )


def test_unknown_experiment_raises_error():
    try:
        get_ablation_columns("not_a_real_experiment")
    except ValueError as exc:
        assert "Unknown ablation experiment" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for unknown experiment."
        )


def test_cross_validate_ablation_returns_metrics():
    X, y = make_sample_dataframe(
        n_samples=20
    )

    result = cross_validate_ablation(
        X,
        y,
        "profile_growth",
        n_splits=5,
        random_state=42,
    )

    assert result["experiment"] == "profile_growth"
    assert result["n_features"] > 0
    assert result["n_splits"] == 5

    assert np.isfinite(result["mae_mean"])
    assert np.isfinite(result["mae_std"])

    assert np.isfinite(result["rmse_mean"])
    assert np.isfinite(result["rmse_std"])

    assert np.isfinite(result["r2_mean"])
    assert np.isfinite(result["r2_std"])


def test_cross_validation_returns_one_result_per_fold():
    X, y = make_sample_dataframe(
        n_samples=20
    )

    result = cross_validate_ablation(
        X,
        y,
        "optical_only",
        n_splits=5,
        random_state=42,
    )

    assert len(result["fold_mae"]) == 5
    assert len(result["fold_rmse"]) == 5
    assert len(result["fold_r2"]) == 5


def test_cross_validation_is_reproducible():
    X, y = make_sample_dataframe(
        n_samples=20
    )

    result_a = cross_validate_ablation(
        X,
        y,
        "all_modalities",
        n_splits=5,
        random_state=42,
    )

    result_b = cross_validate_ablation(
        X,
        y,
        "all_modalities",
        n_splits=5,
        random_state=42,
    )

    assert result_a["mae_mean"] == result_b["mae_mean"]
    assert result_a["rmse_mean"] == result_b["rmse_mean"]
    assert result_a["r2_mean"] == result_b["r2_mean"]


def test_cross_validation_rejects_invalid_split_count():
    X, y = make_sample_dataframe(
        n_samples=20
    )

    try:
        cross_validate_ablation(
            X,
            y,
            "profile_growth",
            n_splits=1,
        )
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for n_splits < 2."
        )