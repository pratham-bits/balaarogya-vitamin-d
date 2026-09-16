from __future__ import annotations

import joblib
import pytest

from src.models.artifact_builder import (
    save_model_artifact,
)
from src.models.artifact_metadata import (
    ModelArtifactMetadata,
)


class DummyModel:
    pass


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


def test_save_model_artifact(tmp_path):
    artifact_path = (
        tmp_path / "models" / "vitamin_d_model.joblib"
    )

    model = DummyModel()
    metadata = make_metadata()

    result = save_model_artifact(
        model=model,
        metadata=metadata,
        artifact_path=artifact_path,
    )

    assert result == artifact_path
    assert artifact_path.exists()

    artifact = joblib.load(artifact_path)

    assert isinstance(
        artifact["model"],
        DummyModel,
    )

    assert artifact["metadata"] == metadata


def test_save_model_artifact_creates_parent_directories(
    tmp_path,
):
    artifact_path = (
        tmp_path
        / "nested"
        / "models"
        / "model.joblib"
    )

    save_model_artifact(
        model=DummyModel(),
        metadata=make_metadata(),
        artifact_path=artifact_path,
    )

    assert artifact_path.exists()


def test_save_model_artifact_rejects_none_model(
    tmp_path,
):
    artifact_path = tmp_path / "model.joblib"

    with pytest.raises(
        ValueError,
        match="model must not be None",
    ):
        save_model_artifact(
            model=None,
            metadata=make_metadata(),
            artifact_path=artifact_path,
        )


def test_save_model_artifact_rejects_invalid_metadata(
    tmp_path,
):
    artifact_path = tmp_path / "model.joblib"

    with pytest.raises(
        TypeError,
        match="metadata must be a ModelArtifactMetadata",
    ):
        save_model_artifact(
            model=DummyModel(),
            metadata={"validation_status": "development_only"},
            artifact_path=artifact_path,
        )