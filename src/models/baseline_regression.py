"""Leakage-safe baseline regression pipeline for the NHANES development dataset."""

from __future__ import annotations
import numpy as np
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "LBXVIDMS"
ID_COLUMN = "SEQN"

NUMERIC_FEATURES = ["RIDAGEYR", "BMXWT", "DR1TVD"]
CATEGORICAL_FEATURES = ["RIAGENDR", "DBQ197"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_analytical_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Analytical dataset not found: {path}")

    df = pd.read_csv(path)

    required = [ID_COLUMN, TARGET, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return df.copy()


def make_preprocessor(
    *,
    include_missing_indicator: bool = False,
    exclude_dr1tvd: bool = False,
) -> ColumnTransformer:
    numeric_features = ["RIDAGEYR", "BMXWT"]
    if not exclude_dr1tvd:
        numeric_features.append("DR1TVD")

    numeric_steps = [
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=include_missing_indicator,
        )),
        ("scaler", StandardScaler()),
    ]

    categorical_features = CATEGORICAL_FEATURES

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(
    estimator,
    *,
    include_missing_indicator: bool = False,
    exclude_dr1tvd: bool = False,
) -> Pipeline:
    return Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                include_missing_indicator=include_missing_indicator,
                exclude_dr1tvd=exclude_dr1tvd,
            ),
        ),
        ("model", estimator),
    ])


def get_baseline_models(random_state: int = 42) -> dict:
    return {
        "Dummy Mean": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state,
        ),
    }


def evaluate_baseline_models(
    df: pd.DataFrame,
    *,
    include_missing_indicator: bool = False,
    exclude_dr1tvd: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Run 5-fold CV on training data and evaluate the selected models on an
    untouched 20% test set.

    Returns:
        results_df, fitted_models
    """
    work = df.dropna(subset=[TARGET]).copy()

    selected_features = ["RIDAGEYR", "BMXWT", "RIAGENDR", "DBQ197"]
    if not exclude_dr1tvd:
        selected_features.insert(2, "DR1TVD")

    X = work[selected_features]
    y = work[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    rows = []
    fitted_models = {}

    for model_name, estimator in get_baseline_models(random_state).items():
        pipeline = make_model_pipeline(
            estimator,
            include_missing_indicator=include_missing_indicator,
            exclude_dr1tvd=exclude_dr1tvd,
        )

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append({
            "model": model_name,
            "cv_mae": -cv_scores["test_mae"].mean(),
            "cv_rmse": -cv_scores["test_rmse"].mean(),
            "cv_r2": cv_scores["test_r2"].mean(),
            "test_mae": mean_absolute_error(y_test, predictions),
            "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "test_r2": r2_score(y_test, predictions),
            "test_size": len(y_test),
            "train_size": len(y_train),
        })

        fitted_models[model_name] = pipeline

    return pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True), fitted_models


def run_all_missingness_experiments(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare the three pre-specified DR1TVD strategies."""
    experiment_rows = []

    strategies = [
        ("median_imputation", False, False),
        ("median_imputation_plus_missing_indicator", True, False),
        ("exclude_DR1TVD", False, True),
    ]

    for strategy, indicator, exclude in strategies:
        result, _ = evaluate_baseline_models(
            df,
            include_missing_indicator=indicator,
            exclude_dr1tvd=exclude,
            random_state=random_state,
        )
        result.insert(0, "missingness_strategy", strategy)
        experiment_rows.append(result)

    return pd.concat(experiment_rows, ignore_index=True)
