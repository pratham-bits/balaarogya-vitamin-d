from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from api.schemas.inference import (
    VitaminDInferenceRequest,
    VitaminDInferenceResponse,
)
from src.data.india_schema import IndianVitaminDRecord
from src.features.multimodal_assembler import (
    assemble_multimodal_features,
)
from src.models.artifact_metadata import (
    ModelArtifactMetadata,
)
from src.optical.pipeline import (
    assess_optical_image,
    load_image,
)
from src.optical.schema import OpticalAssessment


OpticalAssessmentProvider = Callable[
    [str],
    OpticalAssessment,
]


class VitaminDInferenceService:
    """
    Orchestrates Vitamin-D inference.

    The service does not contain clinical thresholds or clinical
    decision rules. Those belong to a separately validated model/
    decision layer.

    If model artifact metadata is available, the service also enforces
    the age range represented by that training artifact.
    """

    def __init__(
        self,
        model: Any | None = None,
        optical_provider: OpticalAssessmentProvider | None = None,
        model_version: str = "unvalidated-prototype",
        metadata: ModelArtifactMetadata | None = None,
    ) -> None:
        self.model = model
        self.optical_provider = optical_provider
        self.model_version = model_version
        self.metadata = metadata

    def assess(
        self,
        request: VitaminDInferenceRequest,
    ) -> VitaminDInferenceResponse:
        """
        Assess one child.

        If no validated model is available, return an explicit
        unable_to_assess response rather than generating a fabricated
        prediction.

        If artifact metadata is available, the child's age must fall
        within the training age range recorded in that metadata.
        """

        if not isinstance(
            request,
            VitaminDInferenceRequest,
        ):
            raise TypeError(
                "request must be a VitaminDInferenceRequest."
            )

        if self.model is None:
            return VitaminDInferenceResponse(
                child_id=request.child_id,
                session_id=request.session_id,
                assessment_status="unable_to_assess",
                message=(
                    "A validated Vitamin-D model is not available "
                    "for inference."
                ),
            )

        age_validation_response = (
            self._validate_training_age_range(request)
        )

        if age_validation_response is not None:
            return age_validation_response

        record = self._build_record(request)

        optical_assessment = self._get_optical_assessment(
            request
        )

        optical_status = self._get_optical_status(
            optical_assessment
        )

        # If an image was explicitly provided but the optical
        # pipeline could not produce usable features, do not
        # continue with a misleading assessment.
        if (
            request.image_path is not None
            and optical_status != "acceptable"
        ):
            return VitaminDInferenceResponse(
                child_id=request.child_id,
                session_id=request.session_id,
                assessment_status="unable_to_assess",
                optical_status=optical_status,
                message=(
                    "The captured image could not be used for optical "
                    "assessment. Please capture another image with "
                    "better lighting, focus, and visible skin."
                ),
            )

        # Build the canonical Indian multimodal representation.
        # This remains the intended future model input.
        multimodal_features = assemble_multimodal_features(
            record,
            optical_assessment=optical_assessment,
        )

        # Prevent unused-variable warnings while keeping the
        # canonical multimodal representation available for the
        # future multimodal model.
        _ = multimodal_features

        # The currently loaded development artifact is an
        # NHANES-trained, API-compatible 3-feature baseline.
        #
        # It does NOT use the complete multimodal representation.
        # Therefore, adapt the API request explicitly rather than
        # silently passing 50 columns into a 3-feature model.
        features = self._build_api_baseline_features(
            request
        )

        try:
            prediction = self.model.predict(features)

        except ValueError:
            return VitaminDInferenceResponse(
                child_id=request.child_id,
                session_id=request.session_id,
                assessment_status="unable_to_assess",
                optical_status=optical_status,
                message=(
                    "The loaded model is incompatible with the "
                    "current inference feature contract."
                ),
            )

        prediction_value = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        return VitaminDInferenceResponse(
            child_id=request.child_id,
            session_id=request.session_id,
            assessment_status="assessed",
            optical_status=optical_status,
            predicted_25ohd_nmol_l=prediction_value,
            risk_probability=None,
            risk_category=None,
            uncertainty=None,
            recommendation=None,
            message="Assessment completed successfully.",
        )

    @staticmethod
    def _build_api_baseline_features(
        request: VitaminDInferenceRequest,
    ) -> pd.DataFrame:
        """
        Build the feature contract expected by the current
        NHANES API-compatible development model.

        This adapter is development-only. It does not imply that
        NHANES-trained relationships are clinically validated for
        Indian children.

        Feature mapping:

        age_months -> RIDAGEYR
        weight_kg  -> BMXWT
        male       -> RIAGENDR = 1
        female     -> RIAGENDR = 2
        """

        sex_mapping = {
            "male": 1,
            "female": 2,
        }

        sex_code = sex_mapping.get(
            request.child.sex.lower()
        )

        weight_kg = (
            request.growth.weight_kg
            if request.growth is not None
            else None
        )

        return pd.DataFrame(
            [
                {
                    "RIDAGEYR": (
                        request.child.age_months / 12.0
                    ),
                    "BMXWT": weight_kg,
                    "RIAGENDR": sex_code,
                }
            ]
        )

    @staticmethod
    def _get_optical_status(
        optical_assessment: OpticalAssessment | None,
    ) -> str | None:
        """
        Convert an OpticalAssessment into a frontend-friendly status.

        This describes image-processing usability only.
        It does not represent Vitamin-D risk.
        """

        if optical_assessment is None:
            return None

        if optical_assessment.quality.status != "acceptable":
            return "unusable"

        if not optical_assessment.skin_roi.detected:
            return "skin_not_detected"

        if optical_assessment.features is None:
            return "features_unavailable"

        return "acceptable"

    def _validate_training_age_range(
        self,
        request: VitaminDInferenceRequest,
    ) -> VitaminDInferenceResponse | None:
        """
        Ensure the requested age falls within the age range represented
        by the loaded training artifact.

        Returns None when the age is allowed. Otherwise returns an
        explicit unable_to_assess response.
        """

        if self.metadata is None:
            return None

        age_months = request.child.age_months

        if age_months < self.metadata.training_age_min_months:
            return VitaminDInferenceResponse(
                child_id=request.child_id,
                session_id=request.session_id,
                assessment_status="unable_to_assess",
                message=(
                    "The child's age is below the minimum age "
                    "represented by the training dataset."
                ),
            )

        if age_months > self.metadata.training_age_max_months:
            return VitaminDInferenceResponse(
                child_id=request.child_id,
                session_id=request.session_id,
                assessment_status="unable_to_assess",
                message=(
                    "The child's age is above the maximum age "
                    "represented by the training dataset."
                ),
            )

        return None

    def _get_optical_assessment(
        self,
        request: VitaminDInferenceRequest,
    ) -> OpticalAssessment | None:
        """
        Run the optical pipeline when an image path is provided.

        The optical pipeline performs:

            image loading
            -> quality assessment
            -> skin ROI extraction
            -> color normalization
            -> optical feature extraction

        No clinical Vitamin-D decision is made here.
        """

        if request.image_path is not None:
            image_id = request.image_id or "api_image"

            image = load_image(
                request.image_path
            )

            return assess_optical_image(
                image,
                image_id=image_id,
            )

        if (
            request.image_id is not None
            and self.optical_provider is not None
        ):
            return self.optical_provider(
                request.image_id
            )

        return None

    @staticmethod
    def _build_record(
        request: VitaminDInferenceRequest,
    ) -> IndianVitaminDRecord:
        """
        Convert the API request into the internal Indian data record.

        Clinical ground truth is intentionally absent because this is
        an inference-time record.
        """

        growth = request.growth
        sun = request.sun
        nutrition = request.nutrition
        breastfeeding = request.breastfeeding
        supplementation = request.supplementation
        child = request.child

        return IndianVitaminDRecord(
            # ----------------------------------------------------------
            # Child profile
            # ----------------------------------------------------------
            child_id="api_request",
            assessment_id="api_assessment",
            age_months=child.age_months,
            sex=child.sex,
            state=child.state,
            district=child.district,
            residence_type=child.residence_type,
            season=child.season,
            household_wealth_quintile=(
                child.household_wealth_quintile
            ),

            # ----------------------------------------------------------
            # Growth
            # ----------------------------------------------------------
            height_cm=(
                growth.height_cm
                if growth is not None
                else None
            ),
            weight_kg=(
                growth.weight_kg
                if growth is not None
                else None
            ),
            bmi=None,
            height_for_age_z=None,
            weight_for_age_z=None,
            weight_for_height_z=None,
            bmi_for_age_z=None,

            # ----------------------------------------------------------
            # Sun exposure
            # ----------------------------------------------------------
            outdoor_time_bucket=(
                sun.outdoor_time_bucket
                if sun is not None
                else None
            ),
            outdoor_frequency=(
                sun.outdoor_frequency
                if sun is not None
                else None
            ),
            typical_outdoor_time_of_day=(
                sun.typical_outdoor_time_of_day
                if sun is not None
                else None
            ),
            clothing_coverage=(
                sun.clothing_coverage
                if sun is not None
                else None
            ),
            sun_avoidant_behavior=(
                sun.sun_avoidant_behavior
                if sun is not None
                else None
            ),

            # ----------------------------------------------------------
            # Nutrition
            # ----------------------------------------------------------
            diet_type=(
                nutrition.diet_type
                if nutrition is not None
                else None
            ),
            dietary_diversity=(
                nutrition.dietary_diversity
                if nutrition is not None
                else None
            ),
            vitamin_d_rich_food_frequency=(
                nutrition.vitamin_d_rich_food_frequency
                if nutrition is not None
                else None
            ),
            egg_consumption=(
                nutrition.egg_consumption
                if nutrition is not None
                else None
            ),
            dairy_consumption=(
                nutrition.dairy_consumption
                if nutrition is not None
                else None
            ),
            fortified_food_consumption=(
                nutrition.fortified_food_consumption
                if nutrition is not None
                else None
            ),
            complementary_feeding=(
                nutrition.complementary_feeding
                if nutrition is not None
                else None
            ),

            # ----------------------------------------------------------
            # Breastfeeding / early-life factors
            # ----------------------------------------------------------
            breastfeeding_status=(
                breastfeeding.status
                if breastfeeding is not None
                else None
            ),
            breastfeeding_duration_months=(
                breastfeeding.duration_months
                if breastfeeding is not None
                else None
            ),
            maternal_sun_exposure=(
                breastfeeding.maternal_sun_exposure
                if breastfeeding is not None
                else None
            ),

            # ----------------------------------------------------------
            # Supplementation
            # ----------------------------------------------------------
            vitamin_d_supplement_use=(
                supplementation.vitamin_d_use
                if supplementation is not None
                else None
            ),
            supplement_frequency=(
                supplementation.frequency
                if supplementation is not None
                else None
            ),
            recent_supplement_use=(
                supplementation.recent_use
                if supplementation is not None
                else None
            ),

            # ----------------------------------------------------------
            # Optical
            # ----------------------------------------------------------
            image_id=request.image_id,
            image_quality=None,
            skin_roi_detected=None,
            calibration_applied=None,
            rgb_features=None,
            hsv_features=None,
            lab_features=None,
            pigmentation_features=None,

            # ----------------------------------------------------------
            # Laboratory ground truth
            # ----------------------------------------------------------
            lab_25ohd_nmol_l=None,
            lab_assay_method=None,
            lab_sample_date=None,
        )