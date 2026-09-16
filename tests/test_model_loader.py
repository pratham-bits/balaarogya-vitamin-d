from __future__ import annotations

import joblib
import pytest

from src.models.artifact_metadata import (
    ModelArtifactMetadata,
)
from src.models.model_loader import (
    ModelArtifactLoader,
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


def test_model_loader_loads_artifact_bundle(tmp_path):
    artifact_path = tmp_path / "model.joblib"

    original_model = DummyModel()
    original_metadata = make_metadata()

    joblib.dump(
        {
            "model": original_model,
            "metadata": original_metadata,
        },
        artifact_path,
    )

    loader = ModelArtifactLoader(
        artifact_path
    )

    artifact = loader.load()

    assert isinstance(
        artifact["model"],
        DummyModel,
    )

    assert artifact["metadata"] == original_metadata


def test_model_loader_accepts_metadata_dict(tmp_path):
    artifact_path = tmp_path / "model.joblib"

    metadata = make_metadata()

    joblib.dump(
        {
            "model": DummyModel(),
            "metadata": metadata.to_dict(),
        },
        artifact_path,
    )

    artifact = ModelArtifactLoader(
        artifact_path
    ).load()

    assert artifact["metadata"] == metadata


def test_model_loader_rejects_missing_artifact(tmp_path):
    artifact_path = (
        tmp_path / "missing_model.joblib"
    )

    loader = ModelArtifactLoader(
        artifact_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        loader.load()


def test_model_loader_rejects_directory(tmp_path):
    artifact_path = tmp_path / "model_directory"

    artifact_path.mkdir()

    loader = ModelArtifactLoader(
        artifact_path
    )

    with pytest.raises(
        ValueError,
        match="is not a file",
    ):
        loader.load()


def test_model_loader_rejects_non_bundle(tmp_path):
    artifact_path = tmp_path / "model.joblib"

    joblib.dump(
        DummyModel(),
        artifact_path,
    )

    loader = ModelArtifactLoader(
        artifact_path
    )

    with pytest.raises(
        ValueError,
        match="must be a dictionary bundle",
    ):
        loader.load()


def test_model_loader_rejects_missing_model(tmp_path):
    artifact_path = tmp_path / "model.joblib"

    joblib.dump(
        {
            "metadata": make_metadata(),
        },
        artifact_path,
    )

    loader = ModelArtifactLoader(
        artifact_path
    )

    with pytest.raises(
        ValueError,
        match="missing the 'model' field",
    ):
        loader.load()


def test_model_loader_rejects_missing_metadata(tmp_path):
    artifact_path = tmp_path / "model.joblib"

    joblib.dump(
        {
            "model": DummyModel(),
        },
        artifact_path,
    )

    loader = ModelArtifactLoader(
        artifact_path
    )

    with pytest.raises(
        ValueError,
        match="missing the 'metadata' field",
    ):
        loader.load()