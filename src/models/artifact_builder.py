from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.models.artifact_metadata import ModelArtifactMetadata


def save_model_artifact(
    model: Any,
    metadata: ModelArtifactMetadata,
    artifact_path: str | Path,
) -> Path:
    """
    Save a model and its metadata as a single artifact bundle.
    """

    if model is None:
        raise ValueError("model must not be None.")

    if not isinstance(
        metadata,
        ModelArtifactMetadata,
    ):
        raise TypeError(
            "metadata must be a ModelArtifactMetadata instance."
        )

    path = Path(artifact_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "metadata": metadata,
    }

    joblib.dump(
        artifact,
        path,
    )

    return path