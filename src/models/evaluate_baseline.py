"""Command-line runner for Phase 4.4 baseline experiments."""

from pathlib import Path

from src.models.baseline_regression import (
    load_analytical_dataset,
    run_all_missingness_experiments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "vitamin_d_nhanes_2017_2018.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_regression_results.csv"


def main() -> None:
    df = load_analytical_dataset(DATASET_PATH)
    results = run_all_missingness_experiments(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nPhase 4.4 baseline results\n")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
