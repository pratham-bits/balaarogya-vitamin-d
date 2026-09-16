from src.optical.schema import (
    ImageMetadata,
    OpticalAssessment,
    OpticalFeatures,
    QualityAssessment,
    SkinROI,
)


def test_optical_schema_can_be_constructed():

    assessment = OpticalAssessment(
        image=ImageMetadata(
            image_id="IMG-001",
            width_px=1080,
            height_px=1920,
        ),
        quality=QualityAssessment(
            status="acceptable",
            brightness_score=0.72,
            blur_score=0.81,
        ),
        skin_roi=SkinROI(
            detected=True,
            x=100,
            y=200,
            width=300,
            height=400,
        ),
        features=OpticalFeatures(
            rgb_features=[120.0, 110.0, 100.0],
            hsv_features=[20.0, 0.25, 0.47],
            lab_features=[55.0, 15.0, 20.0],
        ),
    )

    assert assessment.image.image_id == "IMG-001"
    assert assessment.quality.status == "acceptable"
    assert assessment.skin_roi.detected is True
    assert assessment.features is not None


def test_optical_schema_supports_failed_roi():

    assessment = OpticalAssessment(
        image=ImageMetadata(image_id="IMG-002"),
        quality=QualityAssessment(status="rejected"),
        skin_roi=SkinROI(detected=False),
    )

    assert assessment.skin_roi.detected is False
    assert assessment.features is None


def test_calibration_defaults_to_false():

    assessment = OpticalAssessment(
        image=ImageMetadata(image_id="IMG-003"),
        quality=QualityAssessment(status="acceptable"),
        skin_roi=SkinROI(detected=True),
    )

    assert assessment.calibration_applied is False