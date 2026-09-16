from __future__ import annotations

import numpy as np
import pytest

from src.optical.pipeline import assess_optical_image
from src.optical.schema import OpticalAssessment
from src.optical.serialization import optical_assessment_to_dict


def test_optical_assessment_serializes_to_dict():
    """OpticalAssessment should serialize to a plain dictionary."""

    image = np.full(
        (300, 300, 3),
        120,
        dtype=np.uint8,
    )

    assessment = assess_optical_image(
        image,
        image_id="TEST-SERIALIZE-001",
    )

    result = optical_assessment_to_dict(assessment)

    assert isinstance(result, dict)

    assert result["image"]["image_id"] == "TEST-SERIALIZE-001"
    assert result["image"]["width_px"] == 300
    assert result["image"]["height_px"] == 300

    assert "quality" in result
    assert "skin_roi" in result
    assert "features" in result

    assert result["calibration_applied"] is False


def test_serialized_result_contains_no_dataclass_objects():
    """Serialized output should contain only plain Python structures."""

    image = np.full(
        (300, 300, 3),
        120,
        dtype=np.uint8,
    )

    assessment = assess_optical_image(
        image,
        image_id="TEST-SERIALIZE-002",
    )

    result = optical_assessment_to_dict(assessment)

    assert isinstance(result, dict)

    for value in result.values():
        assert not isinstance(value, OpticalAssessment)


def test_serialization_rejects_invalid_input():
    """Serializer should reject objects of the wrong type."""

    with pytest.raises(TypeError):
        optical_assessment_to_dict({"invalid": "input"})