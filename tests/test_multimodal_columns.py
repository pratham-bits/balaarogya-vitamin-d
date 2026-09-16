from __future__ import annotations

from src.features.multimodal_columns import (
    PROFILE_COLUMNS,
    GROWTH_COLUMNS,
    SUN_COLUMNS,
    NUTRITION_COLUMNS,
    BREASTFEEDING_COLUMNS,
    SUPPLEMENT_COLUMNS,
    OPTICAL_COLUMNS,
    multimodal_feature_names,
)


def test_optical_columns_are_namespaced():
    assert len(OPTICAL_COLUMNS) == 18

    assert all(
        name.startswith("optical__")
        for name in OPTICAL_COLUMNS
    )


def test_all_feature_names_are_unique():
    names = multimodal_feature_names()

    assert len(names) == len(set(names))


def test_feature_names_preserve_modality_order():
    names = multimodal_feature_names()

    expected = (
        *PROFILE_COLUMNS,
        *GROWTH_COLUMNS,
        *SUN_COLUMNS,
        *NUTRITION_COLUMNS,
        *BREASTFEEDING_COLUMNS,
        *SUPPLEMENT_COLUMNS,
        *OPTICAL_COLUMNS,
    )

    assert names == expected


def test_multimodal_feature_count():
    names = multimodal_feature_names()

    expected_count = (
        len(PROFILE_COLUMNS)
        + len(GROWTH_COLUMNS)
        + len(SUN_COLUMNS)
        + len(NUTRITION_COLUMNS)
        + len(BREASTFEEDING_COLUMNS)
        + len(SUPPLEMENT_COLUMNS)
        + len(OPTICAL_COLUMNS)
    )

    assert len(names) == expected_count


def test_optical_order_matches_frozen_contract():
    names = multimodal_feature_names()

    optical_names = [
        name.removeprefix("optical__")
        for name in names
        if name.startswith("optical__")
    ]

    from src.optical.feature_contract import OPTICAL_FEATURE_NAMES

    assert tuple(optical_names) == OPTICAL_FEATURE_NAMES