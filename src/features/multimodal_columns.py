from __future__ import annotations

from typing import Final

from src.optical.feature_contract import OPTICAL_FEATURE_NAMES


PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "profile__age_months",
    "profile__sex",
    "profile__state",
    "profile__district",
    "profile__residence_type",
    "profile__season",
    "profile__household_wealth_quintile",
)

GROWTH_COLUMNS: Final[tuple[str, ...]] = (
    "growth__height_cm",
    "growth__weight_kg",
    "growth__bmi",
    "growth__height_for_age_z",
    "growth__weight_for_age_z",
    "growth__weight_for_height_z",
    "growth__bmi_for_age_z",
)

SUN_COLUMNS: Final[tuple[str, ...]] = (
    "sun__outdoor_time_bucket",
    "sun__outdoor_frequency",
    "sun__typical_outdoor_time_of_day",
    "sun__clothing_coverage",
    "sun__sun_avoidant_behavior",
)

NUTRITION_COLUMNS: Final[tuple[str, ...]] = (
    "nutrition__diet_type",
    "nutrition__dietary_diversity",
    "nutrition__vitamin_d_rich_food_frequency",
    "nutrition__egg_consumption",
    "nutrition__dairy_consumption",
    "nutrition__fortified_food_consumption",
    "nutrition__complementary_feeding",
)

BREASTFEEDING_COLUMNS: Final[tuple[str, ...]] = (
    "breastfeeding__status",
    "breastfeeding__duration_months",
    "breastfeeding__maternal_sun_exposure",
)

SUPPLEMENT_COLUMNS: Final[tuple[str, ...]] = (
    "supplement__vitamin_d_use",
    "supplement__frequency",
    "supplement__recent_use",
)

OPTICAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"optical__{name}" for name in OPTICAL_FEATURE_NAMES
)


def multimodal_feature_names() -> tuple[str, ...]:
    """
    Return the complete ordered feature-name contract.

    Optical features are appended last and retain the exact frozen
    ordering defined by the optical feature contract.
    """

    return (
        *PROFILE_COLUMNS,
        *GROWTH_COLUMNS,
        *SUN_COLUMNS,
        *NUTRITION_COLUMNS,
        *BREASTFEEDING_COLUMNS,
        *SUPPLEMENT_COLUMNS,
        *OPTICAL_COLUMNS,
    )