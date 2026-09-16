# Phase 3R.4 — Optical Specification

## Frozen prototype pipeline

`Smartphone Image → Quality Check → Skin ROI → Color Normalization → Optical Features`

### Core prototype components

1. Image quality:
   - resolution
   - blur
   - brightness
   - contrast

2. Skin ROI:
   - HSV + YCrCb candidate mask
   - morphological cleanup
   - minimum usable ROI check

3. Color normalization:
   - simple luminance normalization in LAB

4. Optical features:
   - RGB means/std
   - RGB ratios
   - HSV means
   - LAB means/std

## Important scientific boundary

The thresholds and segmentation rules in this prototype are engineering
heuristics. They are not clinical thresholds and are not validated
measurements of Vitamin-D status.

The optical branch must eventually be validated on a standardized Indian
child smartphone-image dataset paired with laboratory 25(OH)D.

## Deferred

- CNN embeddings
- advanced segmentation
- camera-specific calibration
- color reference card calibration
- multi-site imaging
- clinically validated pigmentation indices

## Failure handling

A poor-quality image or unusable skin ROI must be allowed to return an
`unable_to_assess` state rather than forcing a risk prediction.
