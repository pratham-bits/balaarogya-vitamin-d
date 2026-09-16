"""
Indian Vitamin-D Dataset Schema
================================

Canonical schema for the future Indian paired dataset used by
BalAarogya's Vitamin-D module.

Important:
- This schema defines data structure, not clinical diagnostic rules.
- Clinical thresholds and product risk thresholds are intentionally
  not defined here.
- Optical fields represent image-derived measurements/features and
  should not be interpreted as standalone Vitamin-D biomarkers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChildGrowthInput:
    """Growth measurements for a child assessment."""

    age_months: int
    sex: str

    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    def validate(self) -> "ChildGrowthInput":
        """Validate the growth record and return this object."""

        from src.features.age_growth import validate_growth_record

        validate_growth_record(
            {
                "age_months": self.age_months,
                "sex": self.sex,
                "height_cm": self.height_cm,
                "weight_kg": self.weight_kg,
            }
        )

        return self

    def to_dict(self) -> dict:
        """Return the growth input as a dictionary."""

        return {
            "age_months": self.age_months,
            "sex": self.sex,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
        }


@dataclass
class IndianVitaminDRecord:
    """
    Canonical representation of one child assessment.

    The record combines:
    - Child profile
    - Growth
    - Sun exposure
    - Nutrition
    - Breastfeeding / early-life factors
    - Supplementation
    - Geography / environment
    - Smartphone optical assessment
    - Laboratory ground truth
    """

    # ---------------------------------------------------------
    # Child profile
    # ---------------------------------------------------------

    child_id: str
    assessment_id: str

    age_months: int
    sex: str

    state: Optional[str] = None
    district: Optional[str] = None
    residence_type: Optional[str] = None
    season: Optional[str] = None

    # Research / context variables
    household_wealth_quintile: Optional[str] = None


    # ---------------------------------------------------------
    # Growth
    # ---------------------------------------------------------

    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None

    # Derived growth features
    height_for_age_z: Optional[float] = None
    weight_for_age_z: Optional[float] = None
    weight_for_height_z: Optional[float] = None
    bmi_for_age_z: Optional[float] = None

    # ---------------------------------------------------------
    # Sun exposure
    # ---------------------------------------------------------

    outdoor_time_bucket: Optional[str] = None
    outdoor_frequency: Optional[str] = None
    typical_outdoor_time_of_day: Optional[str] = None
    clothing_coverage: Optional[str] = None
    sun_avoidant_behavior: Optional[str] = None

    # ---------------------------------------------------------
    # Nutrition
    # ---------------------------------------------------------

    diet_type: Optional[str] = None
    dietary_diversity: Optional[str] = None
    vitamin_d_rich_food_frequency: Optional[str] = None
    egg_consumption: Optional[str] = None
    dairy_consumption: Optional[str] = None
    fortified_food_consumption: Optional[str] = None
    complementary_feeding: Optional[str] = None

    # ---------------------------------------------------------
    # Breastfeeding / early-life factors
    # ---------------------------------------------------------

    breastfeeding_status: Optional[str] = None
    breastfeeding_duration_months: Optional[float] = None
    maternal_sun_exposure: Optional[str] = None

    # ---------------------------------------------------------
    # Supplementation
    # ---------------------------------------------------------

    vitamin_d_supplement_use: Optional[str] = None
    supplement_frequency: Optional[str] = None
    recent_supplement_use: Optional[str] = None

    # ---------------------------------------------------------
    # Optical / smartphone assessment
    # ---------------------------------------------------------

    image_id: Optional[str] = None
    image_quality: Optional[str] = None
    skin_roi_detected: Optional[bool] = None
    calibration_applied: Optional[bool] = None

    # Optical feature outputs
    rgb_features: Optional[list[float]] = None
    hsv_features: Optional[list[float]] = None
    lab_features: Optional[list[float]] = None
    pigmentation_features: Optional[list[float]] = None

    # ---------------------------------------------------------
    # Laboratory ground truth
    # ---------------------------------------------------------

    lab_25ohd_nmol_l: Optional[float] = None
    lab_assay_method: Optional[str] = None
    lab_sample_date: Optional[str] = None


# =============================================================
# Canonical field groups
# =============================================================

FIELD_GROUPS = {
    "child_profile": [
        "child_id",
        "assessment_id",
        "age_months",
        "sex",
        "state",
        "district",
        "residence_type",
        "season",
        "household_wealth_quintile",
    ],

    "growth": [
        "height_cm",
        "weight_kg",
        "bmi",
        "height_for_age_z",
        "weight_for_age_z",
        "weight_for_height_z",
        "bmi_for_age_z",
    ],

    "sun_exposure": [
        "outdoor_time_bucket",
        "outdoor_frequency",
        "typical_outdoor_time_of_day",
        "clothing_coverage",
        "sun_avoidant_behavior",
    ],

    "nutrition": [
        "diet_type",
        "dietary_diversity",
        "vitamin_d_rich_food_frequency",
        "egg_consumption",
        "dairy_consumption",
        "fortified_food_consumption",
        "complementary_feeding",
    ],

    "breastfeeding": [
        "breastfeeding_status",
        "breastfeeding_duration_months",
        "maternal_sun_exposure",
    ],

    "supplementation": [
        "vitamin_d_supplement_use",
        "supplement_frequency",
        "recent_supplement_use",
    ],

    "optical": [
        "image_id",
        "image_quality",
        "skin_roi_detected",
        "calibration_applied",
        "rgb_features",
        "hsv_features",
        "lab_features",
        "pigmentation_features",
    ],

    "laboratory": [
        "lab_25ohd_nmol_l",
        "lab_assay_method",
        "lab_sample_date",
    ],
}


# =============================================================
# Required fields for a paired clinical/optical record
# =============================================================

REQUIRED_PAIRED_FIELDS = [
    "child_id",
    "assessment_id",
    "age_months",
    "sex",
    "height_cm",
    "weight_kg",
    "image_id",
    "lab_25ohd_nmol_l",
]


# =============================================================
# Utility functions
# =============================================================

def get_all_fields() -> list[str]:
    """Return all canonical schema field names."""

    fields = []

    for group_fields in FIELD_GROUPS.values():
        fields.extend(group_fields)

    return fields


def get_required_fields() -> list[str]:
    """Return fields required for a paired optical + clinical record."""

    return REQUIRED_PAIRED_FIELDS.copy()


def validate_required_fields(record: dict) -> tuple[bool, list[str]]:
    """
    Check whether a record contains all required fields.

    Returns:
        (is_valid, missing_fields)
    """

    missing_fields = [
        field
        for field in REQUIRED_PAIRED_FIELDS
        if record.get(field) is None
    ]

    return len(missing_fields) == 0, missing_fields