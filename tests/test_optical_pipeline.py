from __future__ import annotations

import numpy as np
import pytest

from src.optical.pipeline import assess_optical_image
from src.optical.schema import OpticalAssessment


def test_pipeline_rejects_invalid_image():
    """Invalid image input should raise an appropriate error."""

    invalid_image = np.zeros((300, 300), dtype=np.uint8)

    with pytest.raises(ValueError):
        assess_optical_image(invalid_image)


def test_pipeline_handles_poor_quality_image():
    """Poor-quality images should not produce optical features."""

    image = np.full(
        (300, 300, 3),
        120,
        dtype=np.uint8,
    )

    result = assess_optical_image(
        image,
        image_id="TEST-POOR-001",
    )

    assert isinstance(result, OpticalAssessment)

    assert result.image.image_id == "TEST-POOR-001"

    assert result.quality.status == "unusable"

    assert result.features is None

    assert result.calibration_applied is False


def test_pipeline_returns_structured_assessment():
    """
    A valid image should always return an OpticalAssessment
    without crashing, even if the skin ROI cannot be detected.
    """

    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(300, 300, 3),
        dtype=np.uint8,
    )

    result = assess_optical_image(
        image,
        image_id="TEST-STRUCTURED-001",
    )

    assert isinstance(result, OpticalAssessment)

    assert result.image.image_id == "TEST-STRUCTURED-001"

    assert result.image.width_px == 300
    assert result.image.height_px == 300

    assert result.quality is not None
    assert result.skin_roi is not None

    assert result.calibration_applied is False


def test_pipeline_rejects_non_array_input():
    """Non-numpy input should raise TypeError."""

    with pytest.raises(TypeError):
        assess_optical_image(
            "not_an_image",
            image_id="TEST-INVALID-001",
        )

def test_pipeline_successful_end_to_end_path():
    """A suitable image should reach optical feature extraction."""

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

    image = np.clip(
        base.astype(float) + noise,
        0,
        255,
    ).astype(np.uint8)

    result = assess_optical_image(
        image,
        image_id="TEST-E2E-001",
    )

    assert isinstance(result, OpticalAssessment)

    assert result.quality.status == "acceptable"

    assert result.skin_roi.detected is True
    assert result.skin_roi.area_fraction is not None
    assert 0.0 < result.skin_roi.area_fraction <= 1.0

    assert result.features is not None

    assert len(result.features.rgb_features) > 0
    assert len(result.features.hsv_features) > 0
    assert len(result.features.lab_features) > 0

    assert result.calibration_applied is False