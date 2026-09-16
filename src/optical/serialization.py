from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schema import OpticalAssessment


def optical_assessment_to_dict(
    assessment: OpticalAssessment,
) -> dict[str, Any]:
    """
    Convert an OpticalAssessment dataclass into a
    JSON-compatible dictionary.
    """

    if not isinstance(assessment, OpticalAssessment):
        raise TypeError(
            "assessment must be an OpticalAssessment."
        )

    return asdict(assessment)