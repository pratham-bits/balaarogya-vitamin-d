"""Color normalization utilities for the optical prototype."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_color(image_or_pixels: np.ndarray) -> np.ndarray:
    """
    Apply simple per-image luminance normalization.

    This is an engineering normalization step, NOT camera calibration and
    NOT a validated medical measurement correction.
    """
    if not isinstance(image_or_pixels, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image_or_pixels.ndim == 2:
        pixels = image_or_pixels
        if pixels.shape[1] != 3:
            raise ValueError("Pixel array must have shape (N, 3).")
        bgr = pixels.reshape(-1, 1, 3)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        target = 128.0
        current = float(np.mean(l_channel))
        if current > 1e-6:
            lab[:, :, 0] = np.clip(
                l_channel * (target / current), 0, 255
            ).astype(np.uint8)

        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return normalized.reshape(-1, 3)

    if image_or_pixels.ndim == 3 and image_or_pixels.shape[2] == 3:
        lab = cv2.cvtColor(image_or_pixels, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        target = 128.0
        current = float(np.mean(l_channel))
        if current > 1e-6:
            lab[:, :, 0] = np.clip(
                l_channel * (target / current), 0, 255
            ).astype(np.uint8)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    raise ValueError("Input must be an image (H,W,3) or pixels (N,3).")
