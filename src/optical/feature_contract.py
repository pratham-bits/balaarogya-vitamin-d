from __future__ import annotations

from typing import Final

from .schema import OpticalAssessment


OPTICAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    # RGB
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "r_g_ratio",
    "r_b_ratio",
    "g_b_ratio",

    # HSV
    "h_mean",
    "s_mean",
    "v_mean",

    # LAB
    "l_mean",
    "a_mean",
    "b_lab_mean",
    "l_std",
    "a_std",
    "b_lab_std",
)


def optical_feature_names() -> tuple[str, ...]:
    """Return the frozen optical feature ordering."""

    return OPTICAL_FEATURE_NAMES


def optical_feature_vector(
    assessment: OpticalAssessment,
) -> list[float] | None:
    """
    Convert an OpticalAssessment into the fixed 18-value
    model-ready optical feature vector.

    Returns None when optical features are unavailable.
    """

    if not isinstance(assessment, OpticalAssessment):
        raise TypeError(
            "assessment must be an OpticalAssessment."
        )

    if assessment.features is None:
        return None

    rgb = assessment.features.rgb_features or []
    hsv = assessment.features.hsv_features or []
    lab = assessment.features.lab_features or []

    vector = [
        *rgb,
        *hsv,
        *lab,
    ]

    if len(vector) != len(OPTICAL_FEATURE_NAMES):
        raise ValueError(
            "Optical feature vector length mismatch: "
            f"expected {len(OPTICAL_FEATURE_NAMES)}, "
            f"got {len(vector)}."
        )

    return [float(value) for value in vector]