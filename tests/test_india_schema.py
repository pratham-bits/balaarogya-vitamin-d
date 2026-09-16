from src.data.india_schema import (
    FIELD_GROUPS,
    get_all_fields,
    get_required_fields,
    validate_required_fields,
)


def test_schema_contains_expected_groups():
    expected_groups = {
        "child_profile",
        "growth",
        "sun_exposure",
        "nutrition",
        "breastfeeding",
        "supplementation",
        "optical",
        "laboratory",
    }

    assert expected_groups.issubset(FIELD_GROUPS.keys())


def test_schema_contains_core_fields():
    fields = get_all_fields()

    expected_fields = [
        "age_months",
        "sex",
        "height_cm",
        "weight_kg",
        "outdoor_time_bucket",
        "diet_type",
        "vitamin_d_supplement_use",
        "image_id",
        "lab_25ohd_nmol_l",
    ]

    for field in expected_fields:
        assert field in fields


def test_required_fields_are_defined():
    required = get_required_fields()

    assert "age_months" in required
    assert "image_id" in required
    assert "lab_25ohd_nmol_l" in required


def test_required_field_validation():
    valid_record = {
        "child_id": "CHILD-001",
        "assessment_id": "ASSESSMENT-001",
        "age_months": 34,
        "sex": "female",
        "height_cm": 92.1,
        "weight_kg": 12.8,
        "image_id": "IMG-001",
        "lab_25ohd_nmol_l": 42.5,
    }

    is_valid, missing = validate_required_fields(valid_record)

    assert is_valid is True
    assert missing == []


def test_missing_required_field_is_detected():
    incomplete_record = {
        "child_id": "CHILD-001",
        "assessment_id": "ASSESSMENT-001",
        "age_months": 34,
        "sex": "female",
        "height_cm": 92.1,
        "weight_kg": 12.8,
        "image_id": "IMG-001",
    }

    is_valid, missing = validate_required_fields(incomplete_record)

    assert is_valid is False
    assert "lab_25ohd_nmol_l" in missing