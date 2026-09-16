from __future__ import annotations
from sklearn.model_selection import KFold
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.features.multimodal_ablations import get_ablation_columns
from src.features.multimodal_preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from src.features.multimodal_preprocessing import (
    build_multimodal_preprocessor,
)


def build_ablation_preprocessor(
    feature_columns: tuple[str, ...],
):
    """
    Build a preprocessing pipeline restricted to one ablation group.

    The preprocessing logic remains identical to the main multimodal
    pipeline; only the selected feature columns change.
    """

    numeric_columns = tuple(
        column
        for column in feature_columns
        if column in NUMERIC_COLUMNS
    )

    categorical_columns = tuple(
        column
        for column in feature_columns
        if column in CATEGORICAL_COLUMNS
    )

    base_preprocessor = build_multimodal_preprocessor()

    transformers = []

    if numeric_columns:
        numeric_transformer = base_preprocessor.named_transformers_ \
            if hasattr(base_preprocessor, "named_transformers_") else None

    # Rebuild the transformer using the same preprocessing contract.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline as SklearnPipeline

    if numeric_columns:
        numeric_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, list(numeric_columns))
        )

    if categorical_columns:
        categorical_pipeline = SklearnPipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(categorical_columns),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_ablation_model(
    experiment_name: str,
) -> Pipeline:
    """
    Build a baseline Linear Regression model for one ablation experiment.
    """

    feature_columns = get_ablation_columns(experiment_name)

    preprocessor = build_ablation_preprocessor(
        feature_columns
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def evaluate_ablation(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    experiment_name: str,
) -> dict[str, Any]:
    """
    Fit and evaluate one ablation experiment.

    Returns MAE, RMSE and R² on the supplied evaluation data.
    """

    feature_columns = get_ablation_columns(experiment_name)

    missing_columns = [
        column
        for column in feature_columns
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_columns)
        )

    X_selected = X.loc[:, feature_columns]

    model = build_ablation_model(experiment_name)

    model.fit(X_selected, y)

    predictions = model.predict(X_selected)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = model.score(X_selected, y)

    return {
        "experiment": experiment_name,
        "n_features": len(feature_columns),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def cross_validate_ablation(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    experiment_name: str,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate one ablation experiment using K-fold cross-validation.

    The preprocessing pipeline is fitted independently inside each
    training fold to prevent preprocessing leakage.

    Returns mean and standard deviation for MAE, RMSE and R².
    """

    feature_columns = get_ablation_columns(experiment_name)

    missing_columns = [
        column
        for column in feature_columns
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_columns)
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    if len(X) < n_splits:
        raise ValueError(
            "Number of samples must be at least n_splits."
        )

    X_selected = X.loc[:, feature_columns]

    y_array = np.asarray(y)

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_r2: list[float] = []

    for train_indices, validation_indices in kfold.split(X_selected):

        X_train = X_selected.iloc[train_indices]
        X_validation = X_selected.iloc[validation_indices]

        y_train = y_array[train_indices]
        y_validation = y_array[validation_indices]

        model = build_ablation_model(
            experiment_name
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        fold_mae.append(
            float(
                mean_absolute_error(
                    y_validation,
                    predictions,
                )
            )
        )

        fold_rmse.append(
            float(
                np.sqrt(
                    mean_squared_error(
                        y_validation,
                        predictions,
                    )
                )
            )
        )

        fold_r2.append(
            float(
                model.score(
                    X_validation,
                    y_validation,
                )
            )
        )

    return {
        "experiment": experiment_name,
        "n_features": len(feature_columns),
        "n_splits": n_splits,
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae, ddof=1)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std": float(np.std(fold_rmse, ddof=1)),
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2, ddof=1)),
        "fold_mae": fold_mae,
        "fold_rmse": fold_rmse,
        "fold_r2": fold_r2,
    }