from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.multimodal_ablations import (
    ablation_experiment_names,
)
from src.features.multimodal_columns import (
    multimodal_feature_names,
)
from src.models.multimodal_ablation import (
    cross_validate_ablation,
)


def validate_ablation_dataset(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> None:
    """
    Validate the dataset before running ablation experiments.

    The dataset must contain the complete frozen multimodal feature
    contract. The laboratory 25(OH)D value is supplied separately
    as y.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    required_columns = multimodal_feature_names()

    missing_columns = [
        column
        for column in required_columns
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required multimodal columns: "
            + ", ".join(missing_columns)
        )

    y_array = np.asarray(y)

    if not np.isfinite(y_array).all():
        raise ValueError(
            "Target y contains non-finite values."
        )


def run_multimodal_ablation_study(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run all frozen multimodal ablation experiments.

    This function is intended for genuine development/validation
    datasets containing the complete multimodal feature contract.

    Synthetic data must not be used as evidence of model performance.

    Returns
    -------
    pandas.DataFrame
        One row per ablation experiment with mean and standard
        deviation for MAE, RMSE and R².
    """

    validate_ablation_dataset(X, y)

    results: list[dict[str, Any]] = []

    for experiment_name in ablation_experiment_names():

        result = cross_validate_ablation(
            X,
            y,
            experiment_name,
            n_splits=n_splits,
            random_state=random_state,
        )

        results.append(
            {
                "experiment": result["experiment"],
                "n_features": result["n_features"],
                "n_splits": result["n_splits"],
                "mae_mean": result["mae_mean"],
                "mae_std": result["mae_std"],
                "rmse_mean": result["rmse_mean"],
                "rmse_std": result["rmse_std"],
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
            }
        )

    return pd.DataFrame(results)
