"""RGB/HSV/LAB optical feature extraction."""

from __future__ import annotations

import cv2
import numpy as np


def extract_optical_features(bgr_pixels: np.ndarray) -> dict[str, float]:
    """
    Extract summary color features from skin ROI pixels.

    OpenCV uses BGR ordering. Feature names below are reported in conventional
    RGB/LAB/HSV terminology after conversion.
    """
    if not isinstance(bgr_pixels, np.ndarray):
        raise TypeError("bgr_pixels must be a NumPy array.")
    if bgr_pixels.ndim != 2 or bgr_pixels.shape[1] != 3:
        raise ValueError("bgr_pixels must have shape (N, 3).")
    if len(bgr_pixels) == 0:
        raise ValueError("bgr_pixels must contain at least one pixel.")

    pixels = np.clip(bgr_pixels, 0, 255).astype(np.uint8)
    image = pixels.reshape(-1, 1, 3)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    r, g, b = rgb.T.astype(np.float64)
    h, s, v = hsv.T.astype(np.float64)
    l_star, a_star, b_star = lab.T.astype(np.float64)

    eps = 1e-6

    return {
        "r_mean": float(np.mean(r)),
        "g_mean": float(np.mean(g)),
        "b_mean": float(np.mean(b)),
        "r_std": float(np.std(r)),
        "g_std": float(np.std(g)),
        "b_std": float(np.std(b)),
        "r_g_ratio": float(np.mean(r) / (np.mean(g) + eps)),
        "r_b_ratio": float(np.mean(r) / (np.mean(b) + eps)),
        "g_b_ratio": float(np.mean(g) / (np.mean(b) + eps)),
        "h_mean": float(np.mean(h)),
        "s_mean": float(np.mean(s)),
        "v_mean": float(np.mean(v)),
        "l_mean": float(np.mean(l_star)),
        "a_mean": float(np.mean(a_star)),
        "b_lab_mean": float(np.mean(b_star)),
        "l_std": float(np.std(l_star)),
        "a_std": float(np.std(a_star)),
        "b_lab_std": float(np.std(b_star)),
    }
