from __future__ import annotations

import cv2
import numpy as np

from .features import extract_optical_features
from .normalization import normalize_color
from .quality import check_image_quality
from .roi import extract_skin_roi
from .schema import (
    ImageMetadata,
    OpticalAssessment,
    OpticalFeatures,
    QualityAssessment,
    SkinROI,
)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk as a BGR OpenCV image."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel BGR image."
        )

    return image


def assess_optical_image(
    image: np.ndarray,
    *,
    image_id: str = "unknown",
) -> OpticalAssessment:
    """
    Run the complete optical preprocessing pipeline.

    Pipeline:
        Image
        -> Quality assessment
        -> Skin ROI extraction
        -> Color normalization
        -> Optical feature extraction

    This function performs optical preprocessing only.
    It does not estimate Vitamin-D deficiency or provide a diagnosis.
    """

    # ---------------------------------------------------------
    # 1. Validate image
    # ---------------------------------------------------------
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel BGR image."
        )

    height, width = image.shape[:2]

    # ---------------------------------------------------------
    # 2. Image metadata
    # ---------------------------------------------------------
    image_metadata = ImageMetadata(
        image_id=image_id,
        width_px=width,
        height_px=height,
    )

    # ---------------------------------------------------------
    # 3. Quality assessment
    # ---------------------------------------------------------
    quality_result = check_image_quality(image)

    quality_assessment = QualityAssessment(
        status=(
            "acceptable"
            if quality_result["usable"]
            else "unusable"
        ),
        brightness_score=quality_result.get("brightness"),
        blur_score=quality_result.get("blur_score"),
        exposure_score=None,
        quality_flags=quality_result.get("reasons", []),
    )

    # ---------------------------------------------------------
    # 4. Stop if image quality is unacceptable
    # ---------------------------------------------------------
    if not quality_result["usable"]:
        return OpticalAssessment(
            image=image_metadata,
            quality=quality_assessment,
            skin_roi=SkinROI(
                detected=False,
            ),
            features=None,
            calibration_applied=False,
        )

    # ---------------------------------------------------------
    # 5. Skin ROI extraction
    # ---------------------------------------------------------
    skin_pixels, roi_result = extract_skin_roi(image)

    skin_roi = SkinROI(
        detected=roi_result.get("usable", False),
        area_fraction=roi_result.get("coverage_ratio"),
    )

    # ---------------------------------------------------------
    # 6. Stop if skin ROI is unavailable
    # ---------------------------------------------------------
    if (
        not roi_result.get("usable", False)
        or skin_pixels is None
        or skin_pixels.size == 0
    ):
        return OpticalAssessment(
            image=image_metadata,
            quality=quality_assessment,
            skin_roi=skin_roi,
            features=None,
            calibration_applied=False,
        )

    # ---------------------------------------------------------
    # 7. Color normalization
    # ---------------------------------------------------------
    normalized_pixels = normalize_color(skin_pixels)

    # ---------------------------------------------------------
    # 8. Optical feature extraction
    # ---------------------------------------------------------
    feature_result = extract_optical_features(
        normalized_pixels
    )

    # ---------------------------------------------------------
    # 9. Convert extracted features into schema format
    # ---------------------------------------------------------
    rgb_features = []
    hsv_features = []
    lab_features = []

    for key, value in feature_result.items():

        if key.startswith(("r_", "g_", "b_")) or key in {
            "r_g_ratio",
            "r_b_ratio",
            "g_b_ratio",
        }:
            rgb_features.append(float(value))

        elif key.startswith(("h_", "s_", "v_")):
            hsv_features.append(float(value))

        elif key.startswith(("l_", "a_", "b_lab_")):
            lab_features.append(float(value))

    # No separately validated pigmentation feature vector
    # exists in the current implementation.
    pigmentation_features = []

    optical_features = OpticalFeatures(
        rgb_features=rgb_features,
        hsv_features=hsv_features,
        lab_features=lab_features,
        pigmentation_features=pigmentation_features,
    )

    # ---------------------------------------------------------
    # 10. Return structured assessment
    # ---------------------------------------------------------
    return OpticalAssessment(
        image=image_metadata,
        quality=quality_assessment,
        skin_roi=skin_roi,
        features=optical_features,
        calibration_applied=False,
    )