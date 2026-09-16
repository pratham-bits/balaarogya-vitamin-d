from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.india_schema import IndianVitaminDRecord
from src.optical.feature_contract import OPTICAL_FEATURE_NAMES
from src.optical.schema import OpticalAssessment
from src.optical.feature_contract import optical_feature_vector

from .multimodal_columns import multimodal_feature_names


def assemble_multimodal_features(
    record: IndianVitaminDRecord,
    optical_assessment: OpticalAssessment | None = None,
) -> pd.DataFrame:
    """
    Assemble one child record into a single multimodal feature row.

    The returned DataFrame uses the frozen namespaced multimodal
    feature-column contract.

    Optical features are included only when a valid OpticalAssessment
    is supplied. Otherwise, optical columns are represented as NaN.
    """

    if not isinstance(record, IndianVitaminDRecord):
        raise TypeError("record must be an IndianVitaminDRecord.")

    if optical_assessment is not None and not isinstance(
        optical_assessment, OpticalAssessment
    ):
        raise TypeError(
            "optical_assessment must be an OpticalAssessment or None."
        )

    row: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------
    row["profile__age_months"] = record.age_months
    row["profile__sex"] = record.sex
    row["profile__state"] = record.state
    row["profile__district"] = record.district
    row["profile__residence_type"] = record.residence_type
    row["profile__season"] = record.season
    row["profile__household_wealth_quintile"] = (
        record.household_wealth_quintile
    )

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------
    row["growth__height_cm"] = record.height_cm
    row["growth__weight_kg"] = record.weight_kg
    row["growth__bmi"] = record.bmi
    row["growth__height_for_age_z"] = record.height_for_age_z
    row["growth__weight_for_age_z"] = record.weight_for_age_z
    row["growth__weight_for_height_z"] = record.weight_for_height_z
    row["growth__bmi_for_age_z"] = record.bmi_for_age_z

    # ------------------------------------------------------------------
    # Sun exposure
    # ------------------------------------------------------------------
    row["sun__outdoor_time_bucket"] = record.outdoor_time_bucket
    row["sun__outdoor_frequency"] = record.outdoor_frequency
    row["sun__typical_outdoor_time_of_day"] = (
        record.typical_outdoor_time_of_day
    )
    row["sun__clothing_coverage"] = record.clothing_coverage
    row["sun__sun_avoidant_behavior"] = record.sun_avoidant_behavior

    # ------------------------------------------------------------------
    # Nutrition
    # ------------------------------------------------------------------
    row["nutrition__diet_type"] = record.diet_type
    row["nutrition__dietary_diversity"] = record.dietary_diversity
    row["nutrition__vitamin_d_rich_food_frequency"] = (
        record.vitamin_d_rich_food_frequency
    )
    row["nutrition__egg_consumption"] = record.egg_consumption
    row["nutrition__dairy_consumption"] = record.dairy_consumption
    row["nutrition__fortified_food_consumption"] = (
        record.fortified_food_consumption
    )
    row["nutrition__complementary_feeding"] = (
        record.complementary_feeding
    )

    # ------------------------------------------------------------------
    # Breastfeeding
    # ------------------------------------------------------------------
    row["breastfeeding__status"] = record.breastfeeding_status
    row["breastfeeding__duration_months"] = (
        record.breastfeeding_duration_months
    )
    row["breastfeeding__maternal_sun_exposure"] = (
        record.maternal_sun_exposure
    )

    # ------------------------------------------------------------------
    # Supplementation
    # ------------------------------------------------------------------
    row["supplement__vitamin_d_use"] = record.vitamin_d_supplement_use
    row["supplement__frequency"] = record.supplement_frequency
    row["supplement__recent_use"] = record.recent_supplement_use

    # ------------------------------------------------------------------
    # Optical
    # ------------------------------------------------------------------
    optical_vector = None

    if optical_assessment is not None:
        optical_vector = optical_feature_vector(optical_assessment)

    if optical_vector is None:
        for feature_name in OPTICAL_FEATURE_NAMES:
            row[f"optical__{feature_name}"] = float("nan")
    else:
        for feature_name, value in zip(
            OPTICAL_FEATURE_NAMES,
            optical_vector,
            strict=True,
        ):
            row[f"optical__{feature_name}"] = value

    # ------------------------------------------------------------------
    # Final column contract
    # ------------------------------------------------------------------
    columns = multimodal_feature_names()

    return pd.DataFrame(
        [[row[column] for column in columns]],
        columns=columns,
    )