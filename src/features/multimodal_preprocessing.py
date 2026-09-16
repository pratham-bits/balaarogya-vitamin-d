from __future__ import annotations

from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .multimodal_columns import (
    PROFILE_COLUMNS,
    GROWTH_COLUMNS,
    SUN_COLUMNS,
    NUTRITION_COLUMNS,
    BREASTFEEDING_COLUMNS,
    SUPPLEMENT_COLUMNS,
    OPTICAL_COLUMNS,
)


NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "profile__age_months",
    "growth__height_cm",
    "growth__weight_kg",
    "growth__bmi",
    "growth__height_for_age_z",
    "growth__weight_for_age_z",
    "growth__weight_for_height_z",
    "growth__bmi_for_age_z",
    "breastfeeding__duration_months",
    *OPTICAL_COLUMNS,
)


CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "profile__sex",
    "profile__state",
    "profile__district",
    "profile__residence_type",
    "profile__season",
    "profile__household_wealth_quintile",
    "sun__outdoor_time_bucket",
    "sun__outdoor_frequency",
    "sun__typical_outdoor_time_of_day",
    "sun__clothing_coverage",
    "sun__sun_avoidant_behavior",
    "nutrition__diet_type",
    "nutrition__dietary_diversity",
    "nutrition__vitamin_d_rich_food_frequency",
    "nutrition__egg_consumption",
    "nutrition__dairy_consumption",
    "nutrition__fortified_food_consumption",
    "nutrition__complementary_feeding",
    "breastfeeding__status",
    "breastfeeding__maternal_sun_exposure",
    "supplement__vitamin_d_use",
    "supplement__frequency",
    "supplement__recent_use",
)


def preprocessing_column_names() -> tuple[str, ...]:
    return (
        *NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
    )


def build_multimodal_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline for the multimodal feature matrix.

    Numeric:
        median imputation -> standard scaling

    Categorical:
        most-frequent imputation -> one-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_COLUMNS),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )