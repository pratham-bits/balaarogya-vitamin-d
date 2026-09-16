from __future__ import annotations

import numpy as np

from src.data.india_schema import IndianVitaminDRecord
from src.features.multimodal_assembler import (
    assemble_multimodal_features,
)
from src.features.multimodal_columns import (
    multimodal_feature_names,
)
from src.models.multimodal_baseline import (
    build_multimodal_baseline,
)


def make_record() -> IndianVitaminDRecord:
    return IndianVitaminDRecord(
        child_id="test-child",
        assessment_id="test-assessment",
        age_months=36,
        sex="female",
        state="Jharkhand",
        district="Dhanbad",
        residence_type="urban",
        season="summer",
        height_cm=95.0,
        weight_kg=14.0,
        bmi=None,
        height_for_age_z=None,
        weight_for_age_z=None,
        weight_for_height_z=None,
        bmi_for_age_z=None,
        outdoor_time_bucket="30-60min",
        outdoor_frequency="3-4_days_week",
        typical_outdoor_time_of_day="morning",
        clothing_coverage="partial",
        diet_type="vegetarian",
        dietary_diversity="moderate",
        vitamin_d_rich_food_frequency="1-2_times_week",
        egg_consumption="rarely",
        dairy_consumption="daily",
        fortified_food_consumption="rarely",
        breastfeeding_status="previously_breastfed",
        breastfeeding_duration_months=18,
        maternal_sun_exposure="moderate",
        vitamin_d_supplement_use="not_using",
        supplement_frequency="unknown",
        recent_supplement_use="not_using",
        image_id=None,
        image_quality=None,
        skin_roi_detected=None,
        calibration_applied=None,
        rgb_features=None,
        hsv_features=None,
        lab_features=None,
        pigmentation_features=None,
        lab_25ohd_nmol_l=None,
        lab_assay_method=None,
        lab_sample_date=None,
    )


def test_multimodal_assembler_matches_model_contract():
    record = make_record()

    features = assemble_multimodal_features(
        record,
        optical_assessment=None,
    )

    expected_columns = list(
        multimodal_feature_names()
    )

    assert list(features.columns) == expected_columns
    assert features.shape == (
        1,
        len(expected_columns),
    )


def test_multimodal_pipeline_accepts_assembled_features():
    record = make_record()

    features = assemble_multimodal_features(
        record,
        optical_assessment=None,
    )

    model = build_multimodal_baseline()

    # Fit only to establish the preprocessing/model interface.
    # This synthetic data is NOT clinical evidence.
    training_features = features.copy()

    for column in training_features.columns:
        if column.startswith("optical__") or column.startswith("growth__"):
            training_features.loc[0, column] = 1.0

    training_target = np.array([60.0])

    model.fit(
        training_features,
        training_target,
    )

    prediction = model.predict(features)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_missing_optical_features_are_preserved():
    record = make_record()

    features = assemble_multimodal_features(
        record,
        optical_assessment=None,
    )

    optical_columns = [
        column
        for column in features.columns
        if column.startswith("optical__")
    ]

    assert len(optical_columns) == 18
    assert features[optical_columns].isna().all().all()