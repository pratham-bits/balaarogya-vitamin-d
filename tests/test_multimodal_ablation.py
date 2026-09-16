from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_build_ablation_model():
    model = build_ablation_model(
        "profile_growth"
    )

    assert model.__class__.__name__ == "Pipeline"

    step_names = [
        name
        for name, _ in model.steps
    ]

    assert "preprocessor" in step_names
    assert "model" in step_names


def test_ablation_model_fits():
    X, y = make_sample_dataframe()

    model = build_ablation_model(
        "optical_only"
    )

    model.fit(
        X.loc[:, [
            column
            for column in X.columns
            if column.startswith("optical__")
        ]],
        y,
    )


def test_evaluate_ablation_returns_expected_metrics():
    X, y = make_sample_dataframe()

    result = evaluate_ablation(
        X,
        y,
        "profile_growth",
    )

    assert result["experiment"] == "profile_growth"
    assert result["n_features"] > 0

    assert np.isfinite(result["mae"])
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["r2"])


def test_all_modalities_evaluation():
    X, y = make_sample_dataframe()

    result = evaluate_ablation(
        X,
        y,
        "all_modalities",
    )

    assert result["experiment"] == "all_modalities"
    assert result["n_features"] == len(
        multimodal_feature_names()
    )


def test_missing_feature_columns_raise_error():
    X, y = make_sample_dataframe()

    X = X.drop(
        columns=["optical__r_mean"]
    )

    try:
        evaluate_ablation(
            X,
            y,
            "optical_only",
        )
    except ValueError as exc:
        assert "optical__r_mean" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing feature column."
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