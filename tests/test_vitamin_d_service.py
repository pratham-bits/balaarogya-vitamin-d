from __future__ import annotations

import numpy as np

from api.schemas.inference import (
    VitaminDInferenceRequest,
)
from api.services.vitamin_d_service import (
    VitaminDInferenceService,
)
from src.models.artifact_metadata import (
    ModelArtifactMetadata,
)
from src.optical.schema import (
    ImageMetadata,
    OpticalAssessment,
    QualityAssessment,
    SkinROI,
)


class DummyModel:
    def __init__(self) -> None:
        self.predict_called = False

    def predict(self, X):
        self.predict_called = True

        assert X.shape[0] == 1

        return np.array([65.0])


def make_request(
    age_months: int = 36,
) -> VitaminDInferenceRequest:
    return VitaminDInferenceRequest(
        child_id="CHILD-001",
        session_id="SESSION-001",
        child={
            "age_months": age_months,
            "sex": "female",
            "state": "Jharkhand",
            "district": "Dhanbad",
            "residence_type": "urban",
            "season": "winter",
        },
        growth={
            "height_cm": 95.0,
            "weight_kg": 14.0,
        },
        sun={
            "outdoor_time_bucket": "30-60min",
            "outdoor_frequency": "3-4_days_week",
            "typical_outdoor_time_of_day": "morning",
            "clothing_coverage": "partial",
        },
        nutrition={
            "diet_type": "vegetarian",
            "dietary_diversity": "moderate",
            "vitamin_d_rich_food_frequency": "1-2_times_week",
            "egg_consumption": "rarely",
            "dairy_consumption": "daily",
            "fortified_food_consumption": "rarely",
        },
        image_id=None,
    )


def make_metadata() -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_version="v0.1.0",
        target_name="total_25ohd",
        target_unit="nmol/L",
        model_type="linear_regression",
        feature_contract_version="multimodal-v1",
        training_dataset="NHANES 2017-2018",
        validation_status="development_only",
        training_age_min_months=12,
        training_age_max_months=60,
    )


def test_service_without_model_returns_unable_to_assess():
    service = VitaminDInferenceService(
        model=None,
        model_version="test-model",
    )

    response = service.assess(
        make_request()
    )

    assert response.assessment_status == (
        "unable_to_assess"
    )

    assert response.risk_probability is None
    assert response.risk_category is None


def test_service_rejects_invalid_request():
    service = VitaminDInferenceService()

    try:
        service.assess("invalid")
    except TypeError as exc:
        assert "VitaminDInferenceRequest" in str(exc)
    else:
        raise AssertionError(
            "Expected TypeError."
        )


def test_service_uses_model_when_available():
    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        model_version="test-model",
    )

    response = service.assess(
        make_request()
    )

    assert model.predict_called is True
    assert response.assessment_status == "assessed"


def test_service_does_not_return_unvalidated_risk_category():
    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        model_version="test-model",
    )

    response = service.assess(
        make_request()
    )

    assert response.risk_probability is None
    assert response.risk_category is None
    assert response.uncertainty is None
    assert response.recommendation is None


def test_service_handles_missing_optional_modalities():
    model = DummyModel()

    request = VitaminDInferenceRequest(
        child_id="CHILD-001",
        session_id="SESSION-001",
        child={
            "age_months": 24,
            "sex": "male",
        }
    )

    service = VitaminDInferenceService(
        model=model,
        model_version="test-model",
    )

    response = service.assess(request)

    assert response.assessment_status == "assessed"


def test_service_allows_training_age_minimum():
    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        metadata=make_metadata(),
    )

    response = service.assess(
        make_request(age_months=12)
    )

    assert response.assessment_status == "assessed"
    assert model.predict_called is True


def test_service_allows_training_age_maximum():
    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        metadata=make_metadata(),
    )

    response = service.assess(
        make_request(age_months=60)
    )

    assert response.assessment_status == "assessed"
    assert model.predict_called is True


def test_service_rejects_age_below_training_range():
    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        metadata=make_metadata(),
    )

    response = service.assess(
        make_request(age_months=11)
    )

    assert response.assessment_status == "unable_to_assess"
    assert model.predict_called is False
    assert response.risk_probability is None
    assert response.risk_category is None


def test_service_rejects_age_above_training_range():
    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        metadata=make_metadata(),
    )

    response = service.assess(
        make_request(age_months=61)
    )

    assert response.assessment_status == "unable_to_assess"
    assert model.predict_called is False
    assert response.risk_probability is None
    assert response.risk_category is None


def test_service_handles_model_feature_contract_mismatch():
    class IncompatibleModel:
        def predict(self, X):
            raise ValueError(
                "columns are missing: "
                "{'RIDAGEYR', 'BMXWT', 'RIAGENDR', 'DBQ197'}"
            )

    service = VitaminDInferenceService(
        model=IncompatibleModel(),
        model_version="test-model",
    )

    response = service.assess(
        make_request()
    )

    assert response.assessment_status == "unable_to_assess"
    assert response.risk_probability is None
    assert response.risk_category is None
    assert response.uncertainty is None
    assert response.recommendation is None
    assert response.message is not None


def test_api_baseline_feature_adapter_maps_indian_inputs():
    request = make_request(age_months=24)

    features = (
        VitaminDInferenceService
        ._build_api_baseline_features(request)
    )

    assert list(features.columns) == [
        "RIDAGEYR",
        "BMXWT",
        "RIAGENDR",
    ]

    assert features.loc[0, "RIDAGEYR"] == 2.0
    assert features.loc[0, "BMXWT"] == 14.0
    assert features.loc[0, "RIAGENDR"] == 2


def test_service_rejects_unusable_optical_image(monkeypatch):
    """An explicitly supplied unusable image should stop assessment."""

    unusable_assessment = OpticalAssessment(
        image=ImageMetadata(
            image_id="bad_image",
            width_px=100,
            height_px=100,
        ),
        quality=QualityAssessment(
            status="unusable",
            quality_flags=["image_too_blurry"],
        ),
        skin_roi=SkinROI(
            detected=False,
        ),
        features=None,
        calibration_applied=False,
    )

    def fake_load_image(image_path: str):
        return np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

    def fake_assess_optical_image(
        image,
        *,
        image_id: str = "unknown",
    ):
        return unusable_assessment

    monkeypatch.setattr(
        "api.services.vitamin_d_service.load_image",
        fake_load_image,
    )

    monkeypatch.setattr(
        "api.services.vitamin_d_service.assess_optical_image",
        fake_assess_optical_image,
    )

    request = make_request()

    request.image_id = "bad_image"
    request.image_path = "data/raw/bad_image.jpg"

    model = DummyModel()

    service = VitaminDInferenceService(
        model=model,
        model_version="test-model",
    )

    response = service.assess(request)

    assert response.assessment_status == "unable_to_assess"
    assert response.optical_status == "unusable"
    assert response.predicted_25ohd_nmol_l is None
    assert response.message is not None
    assert model.predict_called is False