from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.multimodal_columns import multimodal_feature_names
from src.models.multimodal_baseline import build_multimodal_baseline


def make_training_dataframe(
    n_samples: int = 12,
) -> tuple[pd.DataFrame, np.ndarray]:

    rng = np.random.default_rng(42)

    data: dict[str, list] = {}

    numeric_columns = [
        column
        for column in multimodal_feature_names()
        if (
            column.startswith("profile__age_months")
            or column.startswith("growth__")
            or column.startswith("breastfeeding__duration_months")
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
        60.0
        + 5.0 * X["profile__age_months"].to_numpy()
        + rng.normal(0, 2, n_samples)
    )

    return X, y


def test_multimodal_baseline_is_pipeline():
    model = build_multimodal_baseline()

    assert model.__class__.__name__ == "Pipeline"

    step_names = [name for name, _ in model.steps]

    assert "preprocessor" in step_names
    assert "model" in step_names


def test_multimodal_baseline_fits():
    X, y = make_training_dataframe()

    model = build_multimodal_baseline()

    model.fit(X, y)


def test_multimodal_baseline_predicts():
    X, y = make_training_dataframe()

    model = build_multimodal_baseline()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_multimodal_baseline_handles_missing_values():
    X, y = make_training_dataframe()

    X.loc[0, "growth__height_cm"] = np.nan
    X.loc[1, "optical__r_mean"] = np.nan
    X.loc[2, "nutrition__diet_type"] = None

    model = build_multimodal_baseline()

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_multimodal_baseline_handles_unseen_category():
    X_train, y_train = make_training_dataframe()
    X_test, _ = make_training_dataframe(n_samples=3)

    X_test.loc[:, "profile__season"] = "previously_unseen_season"

    model = build_multimodal_baseline()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == 3
    assert np.isfinite(predictions).all()