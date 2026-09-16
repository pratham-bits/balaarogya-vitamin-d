from __future__ import annotations

import numpy as np

from .feature_contract import optical_feature_vector
from .schema import OpticalAssessment


def get_optical_model_input(
    assessment: OpticalAssessment,
) -> np.ndarray | None:
    """
    Convert one OpticalAssessment into a model-ready feature vector.

    Returns
    -------
    numpy.ndarray | None
        Shape (18,) when optical features are usable.
        None when the optical assessment is unavailable or unusable.
    """

    vector = optical_feature_vector(assessment)

    if vector is None:
        return None

    return np.asarray(vector, dtype=np.float32)