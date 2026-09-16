from __future__ import annotations

import numpy as np
import pytest

from src.optical import assess_optical_image
from src.optical.model_interface import get_optical_model_input


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


def test_model_input_has_expected_shape_and_dtype():
    assessment = assess_optical_image(
        make_good_test_image(),
        image_id="MODEL-001",
    )

    X = get_optical_model_input(assessment)

    assert X is not None
    assert X.shape == (18,)
    assert X.dtype == np.float32


def test_model_input_preserves_feature_values():
    assessment = assess_optical_image(
        make_good_test_image(),
        image_id="MODEL-002",
    )

    X = get_optical_model_input(assessment)

    assert X is not None

    expected = np.asarray(
        [
            *assessment.features.rgb_features,
            *assessment.features.hsv_features,
            *assessment.features.lab_features,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(X, expected)


def test_unusable_optical_assessment_returns_none():
    assessment = assess_optical_image(
        make_bad_test_image(),
        image_id="MODEL-003",
    )

    X = get_optical_model_input(assessment)

    assert X is None


def test_invalid_input_is_rejected():
    with pytest.raises(TypeError):
        get_optical_model_input({"invalid": "assessment"})