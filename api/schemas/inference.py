from __future__ import annotations

# pyright: reportMissingImports=false
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GrowthInput(BaseModel):
    height_cm: Optional[float] = Field(
        default=None,
        gt=0,
    )

    weight_kg: Optional[float] = Field(
        default=None,
        gt=0,
    )


class SunExposureInput(BaseModel):
    outdoor_time_bucket: Optional[str] = None
    outdoor_frequency: Optional[str] = None
    typical_outdoor_time_of_day: Optional[str] = None
    clothing_coverage: Optional[str] = None
    sun_avoidant_behavior: Optional[str] = None


class NutritionInput(BaseModel):
    diet_type: Optional[str] = None
    dietary_diversity: Optional[str] = None
    vitamin_d_rich_food_frequency: Optional[str] = None
    egg_consumption: Optional[str] = None
    dairy_consumption: Optional[str] = None
    fortified_food_consumption: Optional[str] = None
    complementary_feeding: Optional[str] = None


class BreastfeedingInput(BaseModel):
    status: Optional[str] = None

    duration_months: Optional[float] = Field(
        default=None,
        ge=0,
    )

    maternal_sun_exposure: Optional[str] = None


class SupplementInput(BaseModel):
    vitamin_d_use: Optional[str] = None
    frequency: Optional[str] = None
    recent_use: Optional[str] = None


class ChildProfileInput(BaseModel):
    age_months: int = Field(
        ...,
        ge=0,
        le=72,
    )

    sex: str

    state: Optional[str] = None
    district: Optional[str] = None
    residence_type: Optional[str] = None
    season: Optional[str] = None

    # Research / context variables
    household_wealth_quintile: Optional[str] = None
    


class VitaminDInferenceRequest(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    child_id: str = Field(..., alias="childId")
    session_id: str = Field(..., alias="sessionId")

    child: ChildProfileInput

    growth: Optional[GrowthInput] = None

    sun: Optional[SunExposureInput] = None

    nutrition: Optional[NutritionInput] = None

    breastfeeding: Optional[BreastfeedingInput] = None

    supplementation: Optional[SupplementInput] = None

    image_id: Optional[str] = None
    image_path: Optional[str] = None


class VitaminDInferenceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    module: str = "vitascan_nutrition"

    child_id: str = Field(..., alias="childId")
    session_id: str = Field(..., alias="sessionId")

    assessment_status: str

    optical_status: Optional[str] = None

    predicted_25ohd_nmol_l: Optional[float] = Field(
        default=None,
        ge=0,
    )

    # The current regression model does not produce a calibrated probability.
    # Keep this nullable until a proper probability model is implemented.
    risk_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    risk_category: Optional[str] = None

    risk_label: Optional[str] = None

    recommendation: Optional[str] = None

    uncertainty: Optional[dict] = None

    contributing_factors: list[dict] = Field(
        default_factory=list,
    )

    message: Optional[str] = None