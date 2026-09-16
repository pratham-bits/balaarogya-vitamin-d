"""
Optical Assessment Data Contract
=================================

Defines the structured output of the smartphone optical pipeline.

This contract describes image-processing outputs. It does NOT define
Vitamin-D diagnostic rules or clinical thresholds.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageMetadata:
    """Metadata describing the captured smartphone image."""

    image_id: str

    width_px: Optional[int] = None
    height_px: Optional[int] = None

    file_format: Optional[str] = None

    capture_device: Optional[str] = None

    # Capture conditions
    flash_used: Optional[bool] = None
    lighting_condition: Optional[str] = None


@dataclass
class QualityAssessment:
    """Image quality assessment output."""

    status: str

    brightness_score: Optional[float] = None
    blur_score: Optional[float] = None
    exposure_score: Optional[float] = None

    quality_flags: Optional[list[str]] = None


@dataclass
class SkinROI:
    """Detected skin-region information."""

    detected: bool

    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    area_fraction: Optional[float] = None


@dataclass
class OpticalFeatures:
    """
    Image-derived optical features.

    These are computational features and are not automatically
    considered validated biological biomarkers.
    """

    rgb_features: Optional[list[float]] = None
    hsv_features: Optional[list[float]] = None
    lab_features: Optional[list[float]] = None
    pigmentation_features: Optional[list[float]] = None


@dataclass
class OpticalAssessment:
    """Complete output of the optical assessment pipeline."""

    image: ImageMetadata

    quality: QualityAssessment

    skin_roi: SkinROI

    features: Optional[OpticalFeatures] = None

    calibration_applied: bool = False