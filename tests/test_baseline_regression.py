import pandas as pd
import pytest

from src.models.baseline_regression import (
    TARGET,
    load_analytical_dataset,
    make_model_pipeline,
)


def test_dataset_contains_required_columns(tmp_path):
    df = pd.DataFrame({
        "SEQN": [1, 2],
        "RIDAGEYR": [2, 5],
        "RIAGENDR": [1, 2],
        "LBXVIDMS": [60.0, 80.0],
        "BMXWT": [12.0, 18.0],
        "DBQ197": [2, 3],
        "DR1TVD": [4.0, 6.0],
    })

    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)

    loaded = load_analytical_dataset(path)
    assert TARGET in loaded.columns
    assert loaded.shape == (2, 7)


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_analytical_dataset(tmp_path / "missing.csv")


def test_pipeline_fits_without_target_leakage():
    df = pd.DataFrame({
        "RIDAGEYR": [2, 3, 4, 5, 6, 1, 2, 3],
        "RIAGENDR": [1, 2, 1, 2, 1, 2, 1, 2],
        "BMXWT": [12, 14, 16, 18, 20, 10, 13, 15],
        "DBQ197": [1, 2, 3, 3, 2, 1, 3, 2],
        "DR1TVD": [2, 4, 5, 6, 3, 0, 4, 5],
    })
    y = pd.Series([50, 55, 60, 70, 75, 45, 58, 63])

    pipeline = make_model_pipeline(
        estimator=__import__("sklearn").linear_model.LinearRegression()
    )
    pipeline.fit(df, y)
    predictions = pipeline.predict(df)

    assert len(predictions) == len(df)
