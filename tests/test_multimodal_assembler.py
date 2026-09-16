from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.india_schema import IndianVitaminDRecord
from src.features.multimodal_assembler import assemble_multimodal_features
from src.features.multimodal_columns import multimodal_feature_names
from src.optical.pipeline import assess_optical_image


def make_record() -> IndianVitaminDRecord:
    return IndianVitaminDRecord(
        child_id="CHILD001",
        assessment_id="ASSESS001",
        age_months=36,
        sex="female",
        state="Jharkhand",
        district="Dhanbad",
        residence_type="urban",
        season="winter",
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
        breastfeeding_duration_months=18.0,
        maternal_sun_exposure="unknown",
        vitamin_d_supplement_use="not_using",
        supplement_frequency="less_than_weekly",
        recent_supplement_use="not_using",
        image_id="IMG001",
        image_quality="acceptable",
        skin_roi_detected=True,
        calibration_applied=False,
        rgb_features=None,
        hsv_features=None,
        lab_features=None,
        pigmentation_features=None,
        lab_25ohd_nmol_l=70.0,
        lab_assay_method="reference_assay",
        lab_sample_date="2026-09-01",
    )


def make_valid_optical_assessment():
    rng = np.random.default_rng(42)

    base = np.full(
        (300, 300, 3),
        [90, 130, 170],
        dtype=np.uint8,
    )

    noise = rng.normal(
        0,
        55,
        (300, 300, 1),
    )

    image = np.clip(
        base.astype(float) + noise,
        0,
        255,
    ).astype(np.uint8)

    return assess_optical_image(
        image,
        image_id="IMG001",
    )


def test_assembler_returns_expected_columns():
    record = make_record()

    df = assemble_multimodal_features(record)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, len(multimodal_feature_names()))
    assert tuple(df.columns) == multimodal_feature_names()


def test_structured_features_are_assembled_correctly():
    record = make_record()

    df = assemble_multimodal_features(record)

    assert df.loc[0, "profile__age_months"] == 36
    assert df.loc[0, "profile__sex"] == "female"
    assert df.loc[0, "growth__height_cm"] == 95.0
    assert df.loc[0, "growth__weight_kg"] == 14.0
    assert df.loc[0, "sun__outdoor_time_bucket"] == "30-60min"
    assert df.loc[0, "nutrition__diet_type"] == "vegetarian"
    assert df.loc[0, "supplement__vitamin_d_use"] == "not_using"


def test_missing_optical_assessment_is_explicit():
    record = make_record()

    df = assemble_multimodal_features(
        record,
        optical_assessment=None,
    )

    optical_columns = [
        column
        for column in df.columns
        if column.startswith("optical__")
    ]

    assert len(optical_columns) == 18
    assert df[optical_columns].isna().all().all()


def test_valid_optical_assessment_is_assembled():
    record = make_record()
    optical = make_valid_optical_assessment()

    assert optical.features is not None

    df = assemble_multimodal_features(
        record,
        optical_assessment=optical,
    )

    optical_columns = [
        column
        for column in df.columns
        if column.startswith("optical__")
    ]

    assert len(optical_columns) == 18
    assert df[optical_columns].notna().all().all()


def test_optical_values_match_pipeline_output():
    record = make_record()
    optical = make_valid_optical_assessment()

    df = assemble_multimodal_features(
        record,
        optical_assessment=optical,
    )

    expected = optical.features.rgb_features + optical.features.hsv_features + optical.features.lab_features

    actual = df[
        [
            column
            for column in df.columns
            if column.startswith("optical__")
        ]
    ].iloc[0].to_numpy(dtype=float)

    np.testing.assert_allclose(
        actual,
        np.asarray(expected, dtype=float),
    )