from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """
    Metadata describing a Vitamin-D model artifact.

    This metadata is descriptive and safety-oriented. It does not
    establish clinical validity.
    """

    model_version: str
    target_name: str
    target_unit: str
    model_type: str
    feature_contract_version: str
    training_dataset: str
    validation_status: str
    training_age_min_months: int
    training_age_max_months: int

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as a serializable dictionary."""
        return {
            "model_version": self.model_version,
            "target_name": self.target_name,
            "target_unit": self.target_unit,
            "model_type": self.model_type,
            "feature_contract_version": self.feature_contract_version,
            "training_dataset": self.training_dataset,
            "validation_status": self.validation_status,
            "training_age_min_months": self.training_age_min_months,
            "training_age_max_months": self.training_age_max_months,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ModelArtifactMetadata":
        """Create metadata from a dictionary."""

        required_fields = (
            "model_version",
            "target_name",
            "target_unit",
            "model_type",
            "feature_contract_version",
            "training_dataset",
            "validation_status",
            "training_age_min_months",
            "training_age_max_months",
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing_fields:
            raise ValueError(
                "Missing artifact metadata fields: "
                + ", ".join(missing_fields)
            )

        return cls(
            model_version=str(data["model_version"]),
            target_name=str(data["target_name"]),
            target_unit=str(data["target_unit"]),
            model_type=str(data["model_type"]),
            feature_contract_version=str(
                data["feature_contract_version"]
            ),
            training_dataset=str(
                data["training_dataset"]
            ),
            validation_status=str(
                data["validation_status"]
            ),
            training_age_min_months=int(
                data["training_age_min_months"]
            ),
            training_age_max_months=int(
                data["training_age_max_months"]
            ),
        )