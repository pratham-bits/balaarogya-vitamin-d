from pathlib import Path

from sklearn.linear_model import LinearRegression

from src.models.artifact_builder import save_model_artifact
from src.models.artifact_metadata import ModelArtifactMetadata
from src.models.baseline_regression import (
    TARGET,
    load_analytical_dataset,
    make_model_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vitamin_d_nhanes_2017_2018.csv"
)

ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "vitamin_d_baseline.joblib"
)


def main() -> None:
    df = load_analytical_dataset(DATASET_PATH)

    work = df.dropna(subset=[TARGET]).copy()

    selected_features = [
        "RIDAGEYR",
        "BMXWT",
        "RIAGENDR",
        "DBQ197",
    ]

    X = work[selected_features]
    y = work[TARGET]

    model = make_model_pipeline(
        estimator=LinearRegression(),
        exclude_dr1tvd=True,
    )

    model.fit(X, y)

    metadata = ModelArtifactMetadata(
        model_version="nhanes-2017-2018-baseline-v1",
        target_name="LBXVIDMS",
        target_unit="nmol/L",
        model_type="LinearRegression",
        feature_contract_version="baseline-v1",
        training_dataset="NHANES 2017-2018",
        validation_status="development-only-not-india-validated",
        training_age_min_months=12,
        training_age_max_months=72,
    )

    artifact_path = save_model_artifact(
        model=model,
        metadata=metadata,
        artifact_path=ARTIFACT_PATH,
    )

    print(f"Model artifact saved to: {artifact_path}")
    print(f"Training rows: {len(work)}")
    print(f"Features: {selected_features}")
    print(f"Target: {TARGET}")


if __name__ == "__main__":
    main()