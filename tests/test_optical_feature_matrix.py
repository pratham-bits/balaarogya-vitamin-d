from __future__ import annotations

import numpy as np
import pandas as pd

from src.optical import assess_optical_image
from src.optical.feature_contract import OPTICAL_FEATURE_NAMES
from src.optical.feature_matrix import optical_assessments_to_matrix


def make_good_test_image() -> np.ndarray:
    rng = np.random.default_rng(42)

    base = np.full(
        (300, 300, 3),
        [90, 130, 170],
        dtype=np.uint8,
    )

    noise = rng.normal(0, 55, (300, 300, 1))

    return np.clip(
        base.astype(float) + noise,
        0,
        255,
    ).astype(np.uint8)


def make_bad_test_image() -> np.ndarray:
    return np.full(
        (300, 300, 3),
        120,
        dtype=np.uint8,
    )


def test_feature_matrix_has_frozen_columns():
    good = assess_optical_image(
        make_good_test_image(),
        image_id="GOOD-001",
    )

    X, valid_mask = optical_assessments_to_matrix([good])

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == list(OPTICAL_FEATURE_NAMES)
    assert X.shape == (1, 18)
    assert valid_mask.tolist() == [True]


def test_feature_matrix_preserves_feature_values():
    good = assess_optical_image(
        make_good_test_image(),
        image_id="GOOD-002",
    )

    X, valid_mask = optical_assessments_to_matrix([good])

    expected = np.asarray(
        [
            *good.features.rgb_features,
            *good.features.hsv_features,
            *good.features.lab_features,
        ],
        dtype=float,
    )

    assert valid_mask[0]
    np.testing.assert_allclose(
        X.iloc[0].to_numpy(dtype=float),
        expected,
    )


def test_unusable_image_becomes_nan_row():
    good = assess_optical_image(
        make_good_test_image(),
        image_id="GOOD-003",
    )

    bad = assess_optical_image(
        make_bad_test_image(),
        image_id="BAD-001",
    )

    X, valid_mask = optical_assessments_to_matrix([good, bad])

    assert X.shape == (2, 18)
    assert valid_mask.tolist() == [True, False]

    assert X.iloc[1].isna().all()


def test_matrix_preserves_assessment_order():
    first = assess_optical_image(
        make_good_test_image(),
        image_id="FIRST",
    )

    second = assess_optical_image(
        make_good_test_image(),
        image_id="SECOND",
    )

    X, valid_mask = optical_assessments_to_matrix([first, second])

    assert valid_mask.tolist() == [True, True]

    np.testing.assert_allclose(
        X.iloc[0].to_numpy(dtype=float),
        X.iloc[1].to_numpy(dtype=float),
    )


def test_empty_input_returns_empty_matrix():
    X, valid_mask = optical_assessments_to_matrix([])

    assert X.shape == (0, 18)
    assert list(X.columns) == list(OPTICAL_FEATURE_NAMES)
    assert valid_mask.shape == (0,)