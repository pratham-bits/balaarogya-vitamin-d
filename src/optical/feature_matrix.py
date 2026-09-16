from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_contract import OPTICAL_FEATURE_NAMES, optical_feature_vector
from .schema import OpticalAssessment


def optical_assessments_to_matrix(
    assessments: list[OpticalAssessment],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Convert optical assessments into a model-ready feature matrix.

    Returns
    -------
    X : pandas.DataFrame
        One row per assessment with the frozen 18 optical features.
    valid_mask : numpy.ndarray
        Boolean mask indicating which assessments produced usable
        optical features.
    """

    rows: list[list[float]] = []
    valid_mask: list[bool] = []

    for assessment in assessments:
        vector = optical_feature_vector(assessment)

        if vector is None:
            rows.append([np.nan] * len(OPTICAL_FEATURE_NAMES))
            valid_mask.append(False)
        else:
            rows.append(vector)
            valid_mask.append(True)

    X = pd.DataFrame(rows, columns=OPTICAL_FEATURE_NAMES)

    return X, np.asarray(valid_mask, dtype=bool)