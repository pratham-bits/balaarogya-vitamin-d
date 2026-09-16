"""Basic smartphone-image quality checks for the Vitamin-D optical pipeline."""

from __future__ import annotations

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR/RGB 3-channel image.")
    if image.size == 0:
        raise ValueError("image must not be empty.")


def calculate_quality_metrics(image: np.ndarray) -> dict[str, float]:
    """Calculate simple, device-agnostic quality metrics."""
    _validate_image(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(hsv[:, :, 2]))
    contrast = float(np.std(gray))

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
    }


def check_image_quality(
    image: np.ndarray,
    *,
    min_width: int = 224,
    min_height: int = 224,
    min_blur_score: float = 20.0,
    min_brightness: float = 30.0,
    max_brightness: float = 235.0,
    min_contrast: float = 15.0,
) -> dict:
    """
    Apply conservative prototype quality gates.

    Thresholds are engineering defaults, NOT clinical thresholds.
    They must be tuned on the eventual standardized Indian image dataset.
    """
    _validate_image(image)

    h, w = image.shape[:2]
    metrics = calculate_quality_metrics(image)

    reasons = []

    if w < min_width or h < min_height:
        reasons.append("image_resolution_too_low")
    if metrics["blur_score"] < min_blur_score:
        reasons.append("image_too_blurry")
    if metrics["brightness"] < min_brightness:
        reasons.append("image_too_dark")
    if metrics["brightness"] > max_brightness:
        reasons.append("image_too_bright")
    if metrics["contrast"] < min_contrast:
        reasons.append("image_contrast_too_low")

    return {
        "usable": len(reasons) == 0,
        "width": int(w),
        "height": int(h),
        **metrics,
        "reasons": reasons,
    }
