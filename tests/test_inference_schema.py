from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.inference import (
    ChildProfileInput,
    GrowthInput,
    VitaminDInferenceRequest,
    VitaminDInferenceResponse,
)


def test_valid_child_profile():
    child = ChildProfileInput(
        age_months=36,
        sex="female",
        state="Jharkhand",
        district="Dhanbad",
        residence_type="urban",
        season="winter",
    )

    assert child.age_months == 36
    assert child.sex == "female"


def test_age_cannot_exceed_product_range():
    with pytest.raises(ValidationError):
        ChildProfileInput(
            age_months=73,
            sex="female",
        )


def test_negative_growth_values_are_rejected():
    with pytest.raises(ValidationError):
        GrowthInput(
            height_cm=-10,
        )


def test_valid_inference_request():
    request = VitaminDInferenceRequest(
        child_id="CHILD-001",
        session_id="SESSION-001",
        child={
            "age_months": 36,
            "sex": "female",
            "state": "Jharkhand",
        },
        growth={
            "height_cm": 95.0,
            "weight_kg": 14.0,
        },
        image_id="IMG001",
    )

    assert request.child_id == "CHILD-001"
    assert request.session_id == "SESSION-001"
    assert request.child.age_months == 36
    assert request.growth.height_cm == 95.0
    assert request.image_id == "IMG001"


def test_optional_modalities_can_be_missing():
    request = VitaminDInferenceRequest(
        child_id="CHILD-001",
        session_id="SESSION-001",
        child={
            "age_months": 36,
            "sex": "female",
        },
    )

    assert request.growth is None
    assert request.sun is None
    assert request.nutrition is None
    assert request.breastfeeding is None
    assert request.supplementation is None


def test_valid_assessed_response():
    response = VitaminDInferenceResponse(
        child_id="CHILD-001",
    session_id="SESSION-001",
        assessment_status="assessed",
        risk_probability=0.35,
        risk_category="example_category",
        uncertainty="example_uncertainty",
        recommendation="example_recommendation",
    )

    assert response.assessment_status == "assessed"
    assert response.risk_probability == 0.35


def test_probability_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        VitaminDInferenceResponse(
            child_id="CHILD-001",
            session_id="SESSION-001",
            assessment_status="assessed",
            risk_probability=1.5,
        )


def test_unable_to_assess_response():
    response = VitaminDInferenceResponse(
        child_id="CHILD-001",
        session_id="SESSION-001",
        assessment_status="unable_to_assess",
        message="Image quality is insufficient.",
    )

    assert response.assessment_status == "unable_to_assess"
    assert response.risk_probability is None