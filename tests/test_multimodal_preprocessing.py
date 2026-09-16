from __future__ import annotations

from src.features.multimodal_columns import multimodal_feature_names
from src.features.multimodal_preprocessing import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    preprocessing_column_names,
)

import numpy as np
import pandas as pd

from src.features.multimodal_preprocessing import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    build_multimodal_preprocessor,
)

def test_numeric_and_categorical_columns_are_unique():
    combined = (
        *NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
    )

    assert len(combined) == len(set(combined))


def test_preprocessing_columns_match_multimodal_contract():
    assert set(preprocessing_column_names()) == set(
        multimodal_feature_names()
    )


def test_no_column_is_in_both_groups():
    assert set(NUMERIC_COLUMNS).isdisjoint(
        set(CATEGORICAL_COLUMNS)
    )


def test_optical_features_are_numeric():
    optical_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column.startswith("optical__")
    ]

    assert len(optical_columns) == 18


def test_expected_numeric_columns():
    assert "profile__age_months" in NUMERIC_COLUMNS
    assert "growth__height_cm" in NUMERIC_COLUMNS
    assert "growth__weight_kg" in NUMERIC_COLUMNS
    assert "breastfeeding__duration_months" in NUMERIC_COLUMNS


def test_expected_categorical_columns():
    assert "profile__sex" in CATEGORICAL_COLUMNS
    assert "profile__season" in CATEGORICAL_COLUMNS
    assert "sun__outdoor_frequency" in CATEGORICAL_COLUMNS
    assert "nutrition__diet_type" in CATEGORICAL_COLUMNS
    assert "supplement__vitamin_d_use" in CATEGORICAL_COLUMNS

def make_sample_dataframe() -> pd.DataFrame:
    data = {}

    for column in NUMERIC_COLUMNS:
        data[column] = [1.0, 2.0, np.nan]

    for column in CATEGORICAL_COLUMNS:
        data[column] = ["example_a", "example_b", None]

    return pd.DataFrame(data)


def test_preprocessor_is_column_transformer():
    preprocessor = build_multimodal_preprocessor()

    assert preprocessor.__class__.__name__ == "ColumnTransformer"


def test_preprocessor_has_numeric_and_categorical_transformers():
    preprocessor = build_multimodal_preprocessor()

    transformer_names = [
        name
        for name, _, _ in preprocessor.transformers
    ]

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names


def test_preprocessor_fits_and_transforms():
    df = make_sample_dataframe()

    preprocessor = build_multimodal_preprocessor()

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()


def test_numeric_missing_values_are_imputed():
    df = make_sample_dataframe()

    preprocessor = build_multimodal_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert np.isfinite(transformed).all()


def test_unknown_categorical_values_do_not_fail():
    train_df = make_sample_dataframe()

    preprocessor = build_multimodal_preprocessor()
    preprocessor.fit(train_df)

    new_df = make_sample_dataframe()

    for column in CATEGORICAL_COLUMNS:
        new_df.loc[:, column] = "previously_unseen_category"

    transformed = preprocessor.transform(new_df)

    assert transformed.shape[0] == 3
    assert np.isfinite(transformed).all()