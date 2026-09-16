"""
Age and growth feature utilities for BalAarogya.

Product-level canonical representation:
    age_months, sex, height_cm, weight_kg

This module intentionally does NOT calculate WHO z-scores yet.
WHO-standardized growth features will be added only after the exact
reference tables/standardization method are selected and tested.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_GROWTH_COLUMNS = (
    "age_months",
    "sex",
    "height_cm",
    "weight_kg",
)


def validate_age_months(age_months: Any) -> bool:
    """Return True when age is a plausible BalAarogya product age (0-72 months)."""
    if age_months is None or pd.isna(age_months):
        return False
    try:
        value = float(age_months)
    except (TypeError, ValueError):
        return False
    return 0 <= value <= 72


def validate_measurement(value: Any) -> bool:
    """Return True for a positive finite anthropometric measurement."""
    if value is None or pd.isna(value):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return value > 0


def validate_growth_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a product-level growth record.

    Missing height/weight is allowed because community screening may have
    incomplete measurements. Invalid non-missing values are rejected.
    """
    if not validate_age_months(record.get("age_months")):
        raise ValueError("age_months must be between 0 and 72 months.")

    sex = record.get("sex")
    if sex not in {"male", "female", "unknown"}:
        raise ValueError("sex must be 'male', 'female', or 'unknown'.")

    for field in ("height_cm", "weight_kg"):
        value = record.get(field)
        if value is not None and not pd.isna(value) and not validate_measurement(value):
            raise ValueError(f"{field} must be a positive numeric value when provided.")

    return record


def prepare_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate/cast the canonical growth columns without deriving clinical scores.

    Expected columns:
        age_months, sex, height_cm, weight_kg

    Returns a copy with numeric anthropometric columns.
    """
    missing = [c for c in REQUIRED_GROWTH_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required growth columns: {missing}")

    out = df.copy()

    out["age_months"] = pd.to_numeric(out["age_months"], errors="coerce")
    out["height_cm"] = pd.to_numeric(out["height_cm"], errors="coerce")
    out["weight_kg"] = pd.to_numeric(out["weight_kg"], errors="coerce")

    invalid_age = out["age_months"].notna() & ~out["age_months"].between(0, 72)
    if invalid_age.any():
        raise ValueError("Found age_months outside the supported 0-72 month product range.")

    for field in ("height_cm", "weight_kg"):
        invalid = out[field].notna() & (out[field] <= 0)
        if invalid.any():
            raise ValueError(f"Found non-positive values in {field}.")

    return out


def add_derived_growth_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit placeholders for future WHO-standardized features.

    No values are fabricated here. The columns remain NaN until the reference
    standard and implementation are frozen.
    """
    out = prepare_growth_features(df)

    for column in (
        "height_for_age_z",
        "weight_for_age_z",
        "weight_for_height_z",
        "bmi_for_age_z",
    ):
        out[column] = pd.NA

    return out
