"""Prototype skin-region-of-interest extraction."""

from __future__ import annotations

import cv2
import numpy as np


def extract_skin_roi(
    image: np.ndarray,
    *,
    min_pixels: int = 500,
) -> tuple[np.ndarray | None, dict]:
    """
    Extract a conservative candidate skin ROI using HSV/YCrCb masks.

    This is a prototype segmentation step, not a clinically validated
    skin detector. The returned mask should be validated on the target
    Indian-child smartphone dataset.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a 3-channel image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Broad candidate ranges. These are engineering heuristics only.
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 25, 35], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )

    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 125, 70], dtype=np.uint8),
        np.array([255, 185, 135], dtype=np.uint8),
    )

    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    pixel_count = int(np.count_nonzero(mask))

    if pixel_count < min_pixels:
        return None, {
            "usable": False,
            "pixel_count": pixel_count,
            "reason": "insufficient_skin_roi",
        }

    roi = image[mask > 0]

    return roi, {
        "usable": True,
        "pixel_count": pixel_count,
        "coverage_ratio": float(pixel_count / mask.size),
    }
