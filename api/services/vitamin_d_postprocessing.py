from __future__ import annotations

"""
Post-processing layer for the VitaScan Nutrition Vitamin-D assessment.

This module operates AFTER the existing regression model has produced a raw
25(OH)D prediction.

Important:
- Does not train or modify the ML model.
- Does not calculate model feature importance.
- Does not treat optical features as validated biomarkers.
- Converts the raw regression output into the structured screening response.
- Rule-based contributing factors are intentionally isolated so they can be
  replaced by model-derived explanations later.
"""

from dataclasses import dataclass
from typing import Any

from api.schemas.inference import VitaminDInferenceRequest


# ---------------------------------------------------------------------------
# Risk configuration
# ---------------------------------------------------------------------------
#
# These thresholds are explicitly configured for the current prototype
# post-processing layer.
#
# They should be treated as screening/prototype rules rather than as proof
# that the underlying NHANES regression model is clinically validated for
# Indian children.
#

HIGH_RISK_THRESHOLD_NMOL_L = 30.0
MODERATE_RISK_THRESHOLD_NMOL_L = 50.0


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskClassification:
    """Rule-based classification derived from predicted 25(OH)D."""

    category: str
    label: str
    recommendation: str


@dataclass(frozen=True)
class UncertaintyAssessment:
    """Describes completeness of optional contextual input."""

    level: str
    known_field_ratio: float
    message: str


@dataclass(frozen=True)
class PostProcessedPrediction:
    """
    Complete post-processing result.

    risk_probability intentionally remains optional because the current
    regression model does not provide a calibrated probability.
    """

    predicted_25ohd_nmol_l: float
    risk_probability: float | None
    risk_category: str
    risk_label: str
    recommendation: str
    uncertainty: UncertaintyAssessment
    contributing_factors: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Optional contextual fields
# ---------------------------------------------------------------------------
#
# Required model inputs such as age_months, sex, height_cm and weight_kg
# are deliberately NOT included here.
#
# Total:
#   child context         = 5
#   sun exposure          = 5
#   nutrition             = 7
#   breastfeeding         = 3
#   supplementation      = 3
#
# Total optional fields = 23
# ---------------------------------------------------------------------------


OPTIONAL_CONTEXT_FIELDS = (
    # Child context
    "child.state",
    "child.district",
    "child.residence_type",
    "child.season",
    "child.household_wealth_quintile",

    # Sun exposure
    "sun.outdoor_time_bucket",
    "sun.outdoor_frequency",
    "sun.typical_outdoor_time_of_day",
    "sun.clothing_coverage",
    "sun.sun_avoidant_behavior",

    # Nutrition
    "nutrition.diet_type",
    "nutrition.dietary_diversity",
    "nutrition.vitamin_d_rich_food_frequency",
    "nutrition.egg_consumption",
    "nutrition.dairy_consumption",
    "nutrition.fortified_food_consumption",
    "nutrition.complementary_feeding",

    # Breastfeeding
    "breastfeeding.status",
    "breastfeeding.duration_months",
    "breastfeeding.maternal_sun_exposure",

    # Supplementation
    "supplementation.vitamin_d_use",
    "supplementation.frequency",
    "supplementation.recent_use",
)


# ---------------------------------------------------------------------------
# Uncertainty messages
# ---------------------------------------------------------------------------


VERY_HIGH_UNCERTAINTY_MESSAGE = (
    "Prediction based mainly on age, weight, and sex; most contextual "
    "risk factors (sun exposure, diet, geography) are unknown. Confidence "
    "is low — treat as a preliminary flag only."
)

MODERATE_UNCERTAINTY_MESSAGE = (
    "Some contextual risk factors are missing. Recommend collecting "
    "sun-exposure and dietary information for a more reliable assessment."
)

LOW_UNCERTAINTY_MESSAGE = (
    "Most relevant risk factors were provided."
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _get_nested_value(
    request: VitaminDInferenceRequest,
    field_path: str,
) -> Any:
    """
    Safely retrieve a nested request field.

    Example:
        "child.state"
        "sun.outdoor_time_bucket"
    """

    current: Any = request

    for part in field_path.split("."):
        if current is None:
            return None

        current = getattr(current, part, None)

    return current


def _is_known(value: Any) -> bool:
    """
    Return True when an optional field contains meaningful information.

    Unknown values are represented by:
    - None
    - "unknown"
    - empty strings
    """

    if value is None:
        return False

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == "":
            return False

        if normalized == "unknown":
            return False

    return True


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


def classify_risk(predicted_25ohd_nmol_l: float) -> RiskClassification:
    """
    Convert the regression prediction into the configured screening category.

    The raw model prediction is used for threshold comparison. The displayed
    prediction is rounded separately to one decimal place.

    This prevents a rounded value from changing the category at a boundary.
    """

    if predicted_25ohd_nmol_l < HIGH_RISK_THRESHOLD_NMOL_L:
        return RiskClassification(
            category="high_risk",
            label="High Risk — Possible Deficiency",
            recommendation=(
                "Refer for confirmatory serum 25(OH)D test and clinical "
                "assessment."
            ),
        )

    if predicted_25ohd_nmol_l < MODERATE_RISK_THRESHOLD_NMOL_L:
        return RiskClassification(
            category="moderate_risk",
            label="Moderate Risk — Possible Insufficiency",
            recommendation=(
                "Advise increased safe sun exposure and Vitamin D-rich diet; "
                "recommend recheck in 3 months."
            ),
        )

    return RiskClassification(
        category="low_risk",
        label="Low Risk — Likely Sufficient",
        recommendation=(
            "No immediate action needed; continue routine monitoring at "
            "next scheduled visit."
        ),
    )


# ---------------------------------------------------------------------------
# Uncertainty assessment
# ---------------------------------------------------------------------------


def assess_uncertainty(
    request: VitaminDInferenceRequest,
) -> UncertaintyAssessment:
    """
    Calculate uncertainty from completeness of optional contextual inputs.

    There are exactly 23 optional contextual fields.

    known_field_ratio =
        known_fields / total_optional_fields
    """

    total_optional_fields = len(OPTIONAL_CONTEXT_FIELDS)

    known_fields = sum(
        1
        for field_path in OPTIONAL_CONTEXT_FIELDS
        if _is_known(_get_nested_value(request, field_path))
    )

    known_field_ratio = known_fields / total_optional_fields

    # Keep the value stable and clean for JSON output.
    known_field_ratio = round(known_field_ratio, 2)

    if known_field_ratio < 0.3:
        return UncertaintyAssessment(
            level="very_high",
            known_field_ratio=known_field_ratio,
            message=VERY_HIGH_UNCERTAINTY_MESSAGE,
        )

    if known_field_ratio < 0.6:
        return UncertaintyAssessment(
            level="moderate",
            known_field_ratio=known_field_ratio,
            message=MODERATE_UNCERTAINTY_MESSAGE,
        )

    return UncertaintyAssessment(
        level="low",
        known_field_ratio=known_field_ratio,
        message=LOW_UNCERTAINTY_MESSAGE,
    )


# ---------------------------------------------------------------------------
# Rule-based contributing factors
# ---------------------------------------------------------------------------
#
# This function is deliberately isolated from the main post-processing
# function. Later, it can be replaced with model-derived feature importance
# without changing the API response schema.
# ---------------------------------------------------------------------------


def build_contributing_factors(
    request: VitaminDInferenceRequest,
    uncertainty: UncertaintyAssessment,
) -> list[dict[str, Any]]:
    """
    Build a qualitative list of known factors associated with the screening
    output.

    These are contextual associations/rules, NOT model feature importance.
    """

    factors: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Urban residence
    # ---------------------------------------------------------------

    residence_type = _get_nested_value(
        request,
        "child.residence_type",
    )

    if (
        isinstance(residence_type, str)
        and residence_type.strip().lower() == "urban"
    ):
        factors.append(
            {
                "factor": "urban_residence",
                "direction": "increases_risk",
                "note": (
                    "Urban children show higher Vitamin D deficiency "
                    "prevalence in Indian population data (CNNS 2016-18)."
                ),
            }
        )

    # ---------------------------------------------------------------
    # Clothing coverage
    # ---------------------------------------------------------------

    clothing_coverage = _get_nested_value(
        request,
        "sun.clothing_coverage",
    )

    if (
        isinstance(clothing_coverage, str)
        and clothing_coverage.strip().lower()
        in {
            "mostly_covered",
            "fully_covered",
        }
    ):
        factors.append(
            {
                "factor": "clothing_coverage",
                "direction": "increases_risk",
            }
        )

    # ---------------------------------------------------------------
    # Sun-avoidant behaviour
    # ---------------------------------------------------------------

    sun_avoidant_behavior = _get_nested_value(
        request,
        "sun.sun_avoidant_behavior",
    )

    if isinstance(sun_avoidant_behavior, str):
        normalized = sun_avoidant_behavior.strip().lower()

        if normalized in {
            "yes",
            "true",
            "high",
            "highly_avoidant",
        }:
            factors.append(
                {
                    "factor": "sun_avoidant_behavior",
                    "direction": "increases_risk",
                }
            )

    # ---------------------------------------------------------------
    # Vitamin-D supplementation
    # ---------------------------------------------------------------

    vitamin_d_use = _get_nested_value(
        request,
        "supplementation.vitamin_d_use",
    )

    if isinstance(vitamin_d_use, str):
        normalized = vitamin_d_use.strip().lower()

        if normalized == "currently_using":
            factors.append(
                {
                    "factor": "supplementation",
                    "direction": "decreases_risk",
                }
            )

    # ---------------------------------------------------------------
    # Insufficient contextual information
    # ---------------------------------------------------------------

    if uncertainty.known_field_ratio < 0.3:
        factors.append(
            {
                "factor": "insufficient_context_data",
                "direction": "unknown",
                "note": (
                    "Most contextual fields were not provided; this is a "
                    "data-completeness gap, not a model limitation."
                ),
            }
        )

    return factors


# ---------------------------------------------------------------------------
# Main post-processing function
# ---------------------------------------------------------------------------


def postprocess_prediction(
    request: VitaminDInferenceRequest,
    raw_prediction: float,
) -> PostProcessedPrediction:
    """
    Wrap the existing regression prediction with screening-oriented output.

    Parameters
    ----------
    request:
        Original VitaScan inference request.

    raw_prediction:
        Raw 25(OH)D prediction returned by the existing regression model.

    Returns
    -------
    PostProcessedPrediction
        Structured risk category, recommendation, uncertainty and
        contributing factors.

    Notes
    -----
    The underlying regression model is completely untouched.

    `risk_probability` intentionally remains None because the current model
    does not produce a calibrated probability.
    """

    # ---------------------------------------------------------------
    # 1. Validate and round prediction
    # ---------------------------------------------------------------

    prediction_value = float(raw_prediction)

    rounded_prediction = round(prediction_value, 1)

    # ---------------------------------------------------------------
    # 2. Risk classification
    # ---------------------------------------------------------------

    risk = classify_risk(prediction_value)

    # ---------------------------------------------------------------
    # 3. Context completeness / uncertainty
    # ---------------------------------------------------------------

    uncertainty = assess_uncertainty(request)

    # ---------------------------------------------------------------
    # 4. Qualitative contributing factors
    # ---------------------------------------------------------------

    contributing_factors = build_contributing_factors(
        request,
        uncertainty,
    )

    # ---------------------------------------------------------------
    # 5. Return complete post-processed result
    # ---------------------------------------------------------------

    return PostProcessedPrediction(
        predicted_25ohd_nmol_l=rounded_prediction,
        risk_probability=None,
        risk_category=risk.category,
        risk_label=risk.label,
        recommendation=risk.recommendation,
        uncertainty=uncertainty,
        contributing_factors=contributing_factors,
    )