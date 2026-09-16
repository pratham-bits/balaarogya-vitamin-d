from __future__ import annotations

import numpy as np
import pytest

from src.optical import assess_optical_image
from src.optical.feature_contract import (
    OPTICAL_FEATURE_NAMES,
    optical_feature_names,
    optical_feature_vector,
)


def make_test_image() -> np.ndarray:
    """Create a deterministic image that reaches feature extraction."""

    rng = np.random.default_rng(42)

    base = np.full(
        (300, 300, 3),
        [90, 130, 170],
        dtype=np.uint8,
    )

    noise = rng.normal(
        0,
        55,
        (300, 300, 1),
    )

    return np.clip(
        base.astype(float) + noise,
        0,
        255,
    ).astype(np.uint8)


def test_optical_feature_names_are_frozen():

    assert len(OPTICAL_FEATURE_NAMES) == 18

    assert optical_feature_names() == OPTICAL_FEATURE_NAMES

    assert OPTICAL_FEATURE_NAMES[0] == "r_mean"
    assert OPTICAL_FEATURE_NAMES[-1] == "b_lab_std"


def test_optical_feature_vector_has_fixed_length():

    image = make_test_image()

    assessment = assess_optical_image(
        image,
        image_id="TEST-FEATURE-CONTRACT-001",
    )

    vector = optical_feature_vector(assessment)

    assert vector is not None
    assert len(vector) == 18

    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_optical_feature_vector_preserves_feature_order():

    image = make_test_image()

    assessment = assess_optical_image(
        image,
        image_id="TEST-FEATURE-CONTRACT-002",
    )

    vector = optical_feature_vector(assessment)

    assert vector is not None

    expected = [
        *assessment.features.rgb_features,
        *assessment.features.hsv_features,
        *assessment.features.lab_features,
    ]

    assert vector == expected


def test_missing_optical_features_return_none():

    image = np.full(
        (300, 300, 3),
        120,
        dtype=np.uint8,
    )

    assessment = assess_optical_image(
        image,
        image_id="TEST-FEATURE-CONTRACT-003",
    )

    assert assessment.features is None

    assert optical_feature_vector(assessment) is None


def test_feature_vector_rejects_invalid_input():

    with pytest.raises(TypeError):

        optical_feature_vector(
            {"invalid": "assessment"}
        )