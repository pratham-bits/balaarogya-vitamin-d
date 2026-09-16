from src.data.india_questionnaire import (
    QUESTIONNAIRE_VOCABULARY,
    validate_questionnaire_value,
)


def test_questionnaire_vocabulary_contains_core_fields():
    expected_fields = [
        "outdoor_time_bucket",
        "clothing_coverage",
        "diet_type",
        "dietary_diversity",
        "vitamin_d_supplement_use",
        "residence_type",
        "season",
    ]

    for field in expected_fields:
        assert field in QUESTIONNAIRE_VOCABULARY


def test_valid_questionnaire_value():
    assert validate_questionnaire_value(
        "outdoor_time_bucket",
        "30-60min",
    )


def test_invalid_questionnaire_value():
    assert not validate_questionnaire_value(
        "outdoor_time_bucket",
        "30 minutes",
    )


def test_unknown_is_supported():
    assert validate_questionnaire_value(
        "diet_type",
        "unknown",
    )