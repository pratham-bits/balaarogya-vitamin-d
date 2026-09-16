"""
Indian Vitamin-D Questionnaire Specification
==============================================

Controlled vocabulary for the future Indian Vitamin-D dataset.

These categories standardize data collection. They are NOT
clinical risk thresholds and do not contain model weights.
"""


# =============================================================
# Sun exposure
# =============================================================

OUTDOOR_TIME_BUCKETS = [
    "none",
    "<30min",
    "30-60min",
    "1-2h",
    ">2h",
    "unknown",
]

OUTDOOR_FREQUENCY = [
    "rarely",
    "1-2_days_week",
    "3-4_days_week",
    "5-7_days_week",
    "unknown",
]

OUTDOOR_TIME_OF_DAY = [
    "morning",
    "midday",
    "afternoon",
    "mixed",
    "unknown",
]

CLOTHING_COVERAGE = [
    "minimal",
    "partial",
    "mostly_covered",
    "fully_covered",
    "unknown",
]


# =============================================================
# Nutrition
# =============================================================

DIET_TYPES = [
    "vegetarian",
    "non_vegetarian",
    "mixed",
    "unknown",
]

DIETARY_DIVERSITY = [
    "low",
    "moderate",
    "high",
    "unknown",
]

FOOD_FREQUENCY = [
    "never",
    "rarely",
    "1-2_times_week",
    "3-6_times_week",
    "daily",
    "unknown",
]


# =============================================================
# Breastfeeding
# =============================================================

BREASTFEEDING_STATUS = [
    "currently_breastfed",
    "previously_breastfed",
    "never_breastfed",
    "not_applicable",
    "unknown",
]


# =============================================================
# Supplementation
# =============================================================

SUPPLEMENT_USE = [
    "currently_using",
    "recently_used",
    "not_using",
    "unknown",
]

SUPPLEMENT_FREQUENCY = [
    "daily",
    "several_times_week",
    "weekly",
    "less_than_weekly",
    "unknown",
]


# =============================================================
# Context
# =============================================================

RESIDENCE_TYPES = [
    "urban",
    "rural",
    "peri_urban",
    "unknown",
]

SEASONS = [
    "summer",
    "monsoon",
    "autumn",
    "winter",
    "spring",
    "unknown",
]


# =============================================================
# Helper
# =============================================================

QUESTIONNAIRE_VOCABULARY = {
    "outdoor_time_bucket": OUTDOOR_TIME_BUCKETS,
    "outdoor_frequency": OUTDOOR_FREQUENCY,
    "typical_outdoor_time_of_day": OUTDOOR_TIME_OF_DAY,
    "clothing_coverage": CLOTHING_COVERAGE,
    "diet_type": DIET_TYPES,
    "dietary_diversity": DIETARY_DIVERSITY,
    "vitamin_d_rich_food_frequency": FOOD_FREQUENCY,
    "egg_consumption": FOOD_FREQUENCY,
    "dairy_consumption": FOOD_FREQUENCY,
    "fortified_food_consumption": FOOD_FREQUENCY,
    "breastfeeding_status": BREASTFEEDING_STATUS,
    "vitamin_d_supplement_use": SUPPLEMENT_USE,
    "supplement_frequency": SUPPLEMENT_FREQUENCY,
    "recent_supplement_use": SUPPLEMENT_USE,
    "residence_type": RESIDENCE_TYPES,
    "season": SEASONS,
}


def validate_questionnaire_value(
    field_name: str,
    value: str,
) -> bool:
    """Return True if a value belongs to the field's controlled vocabulary."""

    if field_name not in QUESTIONNAIRE_VOCABULARY:
        return False

    return value in QUESTIONNAIRE_VOCABULARY[field_name]