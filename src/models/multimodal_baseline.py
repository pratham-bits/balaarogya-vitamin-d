from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.features.multimodal_preprocessing import (
    build_multimodal_preprocessor,
)


def build_multimodal_baseline() -> Pipeline:
    """
    Build the baseline multimodal regression pipeline.

    Pipeline:
        raw multimodal features
            -> preprocessing
            -> Linear Regression

    The model predicts continuous serum 25(OH)D concentration.
    """

    preprocessor = build_multimodal_preprocessor()

    model = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )