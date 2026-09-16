from src.models.artifact_metadata import (
    ModelArtifactMetadata,
)


def make_metadata():
    return ModelArtifactMetadata(
        model_version="v0.1.0",
        target_name="total_25ohd",
        target_unit="nmol/L",
        model_type="linear_regression",
        feature_contract_version="multimodal-v1",
        training_dataset="NHANES 2017-2018",
        validation_status="development_only",
        training_age_min_months=12,
        training_age_max_months=72,
    )


def test_metadata_to_dict():
    metadata = make_metadata()

    result = metadata.to_dict()

    assert result["model_version"] == "v0.1.0"
    assert result["target_name"] == "total_25ohd"
    assert result["target_unit"] == "nmol/L"
    assert result["model_type"] == "linear_regression"
    assert result["feature_contract_version"] == "multimodal-v1"
    assert result["training_dataset"] == "NHANES 2017-2018"
    assert result["validation_status"] == "development_only"
    assert result["training_age_min_months"] == 12
    assert result["training_age_max_months"] == 72


def test_metadata_round_trip():
    metadata = make_metadata()

    restored = ModelArtifactMetadata.from_dict(
        metadata.to_dict()
    )

    assert restored == metadata


def test_metadata_rejects_missing_fields():
    data = {
        "model_version": "v0.1.0",
    }

    try:
        ModelArtifactMetadata.from_dict(data)
    except ValueError as exc:
        assert "Missing artifact metadata fields" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing metadata fields."
        )