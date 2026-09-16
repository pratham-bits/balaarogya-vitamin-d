import pandas as pd
import pytest  # type: ignore

from src.data.india_schema import ChildGrowthInput
from src.features.age_growth import (
    add_derived_growth_placeholders,
    prepare_growth_features,
    validate_age_months,
)


def test_valid_age_range():
    assert validate_age_months(0)
    assert validate_age_months(36)
    assert validate_age_months(72)


def test_invalid_age_range():
    assert not validate_age_months(-1)
    assert not validate_age_months(73)
    assert not validate_age_months(None)


def test_child_growth_input():
    child = ChildGrowthInput(
        age_months=36,
        sex="female",
        height_cm=94.2,
        weight_kg=13.8,
    )
    assert child.validate().to_dict()["age_months"] == 36


def test_invalid_growth_measurement():
    child = ChildGrowthInput(
        age_months=36,
        sex="female",
        height_cm=94.2,
        weight_kg=-2,
    )
    with pytest.raises(ValueError):
        child.validate()


def test_prepare_growth_features():
    df = pd.DataFrame(
        {
            "age_months": [24, 36],
            "sex": ["male", "female"],
            "height_cm": [85.0, 94.2],
            "weight_kg": [11.2, 13.8],
        }
    )
    out = prepare_growth_features(df)
    assert out["age_months"].dtype.kind in "fi"
    assert out["weight_kg"].tolist() == [11.2, 13.8]


def test_who_placeholders_are_not_fabricated():
    df = pd.DataFrame(
        {
            "age_months": [36],
            "sex": ["female"],
            "height_cm": [94.2],
            "weight_kg": [13.8],
        }
    )
    out = add_derived_growth_placeholders(df)
    assert pd.isna(out.loc[0, "height_for_age_z"])
    assert pd.isna(out.loc[0, "weight_for_age_z"])
    assert pd.isna(out.loc[0, "weight_for_height_z"])
    assert pd.isna(out.loc[0, "bmi_for_age_z"])
