from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.models.artifact_metadata import (
    ModelArtifactMetadata,
)


class ModelArtifactLoader:
    """
    Load a serialized Vitamin-D model artifact bundle.

    The artifact contains:
        - model
        - metadata
    """

    def __init__(
        self,
        artifact_path: str | Path,
    ) -> None:
        self.artifact_path = Path(artifact_path)

    def load(self) -> dict[str, Any]:
        """
        Load and validate the serialized artifact bundle.
        """

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: "
                f"{self.artifact_path}"
            )

        if not self.artifact_path.is_file():
            raise ValueError(
                f"Model artifact path is not a file: "
                f"{self.artifact_path}"
            )

        artifact = joblib.load(self.artifact_path)

        if not isinstance(artifact, dict):
            raise ValueError(
                "Model artifact must be a dictionary bundle."
            )

        if "model" not in artifact:
            raise ValueError(
                "Model artifact is missing the 'model' field."
            )

        if "metadata" not in artifact:
            raise ValueError(
                "Model artifact is missing the 'metadata' field."
            )

        metadata = artifact["metadata"]

        if isinstance(metadata, ModelArtifactMetadata):
            validated_metadata = metadata

        elif isinstance(metadata, dict):
            validated_metadata = (
                ModelArtifactMetadata.from_dict(metadata)
            )

        else:
            raise ValueError(
                "Model artifact metadata must be a "
                "ModelArtifactMetadata object or dictionary."
            )

        return {
            "model": artifact["model"],
            "metadata": validated_metadata,
        }