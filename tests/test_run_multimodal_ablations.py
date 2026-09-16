from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.multimodal_columns import (
    multimodal_feature_names,
)
from src.models.run_multimodal_ablations import (
    validate_ablation_dataset,
    run_multimodal_ablation_study,
)


def make_test_dataframe(
    n_samples: int = 20,
) -> tuple[pd.DataFrame, np.ndarray]:

    rng = np.random.default_rng(42)

    data: dict[str, list] = {}

    for column in multimodal_feature_names():

        if (
            column == "profile__age_months"
            or column.startswith("growth__")
            or column == "breastfeeding__duration_months"
            or column.startswith("optical__")
        ):
            data[column] = rng.normal(
                0,
                1,
                n_samples,
            ).tolist()

        else:
            data[column] = [
                "category_a"
                if i % 2 == 0
                else "category_b"
                for i in range(n_samples)
            ]

    X = pd.DataFrame(data)

    y = (
        50.0
        + 5.0 * X["profile__age_months"].to_numpy()
        + rng.normal(0, 1, n_samples)
    )

    return X, y


def test_validate_ablation_dataset_accepts_valid_data():
    X, y = make_test_dataframe()

    validate_ablation_dataset(X, y)


def test_validate_ablation_dataset_rejects_missing_columns():
    X, y = make_test_dataframe()

    X = X.drop(
        columns=["optical__r_mean"]
    )

    with pytest.raises(ValueError, match="optical__r_mean"):
        validate_ablation_dataset(X, y)


def test_validate_ablation_dataset_rejects_length_mismatch():
    X, y = make_test_dataframe()

    y = y[:-1]

    with pytest.raises(ValueError, match="same number"):
        validate_ablation_dataset(X, y)


def test_validate_ablation_dataset_rejects_non_finite_target():
    X, y = make_test_dataframe()

    y[0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        validate_ablation_dataset(X, y)


def test_ablation_study_returns_expected_experiments():
    X, y = make_test_dataframe()

    results = run_multimodal_ablation_study(
        X,
        y,
        n_splits=5,
        random_state=42,
    )

    assert isinstance(results, pd.DataFrame)

    assert len(results) == 7

    expected_columns = {
        "experiment",
        "n_features",
        "n_splits",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
    }

    assert set(results.columns) == expected_columns


def test_ablation_study_returns_finite_metrics():
    X, y = make_test_dataframe()

    results = run_multimodal_ablation_study(
        X,
        y,
        n_splits=5,
        random_state=42,
    )

    metric_columns = [
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
    ]

    assert np.isfinite(
        results[metric_columns].to_numpy()
    ).all()