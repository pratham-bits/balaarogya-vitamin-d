from pathlib import Path
import json
import pandas as pd


# =============================================================================
# BalAarogya - Vitamin-D Module
# PHASE 3.3 - MISSING DATA & PREPROCESSING
# SUB-PHASE 3.3.6 - FINAL BASELINE DATASET SPECIFICATION
# =============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vitamin_d_nhanes_2017_2018.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_model_specification.json"
)


# =============================================================================
# FROZEN BASELINE SPECIFICATION
# =============================================================================

IDENTIFIER = "SEQN"

TARGET = "LBXVIDMS"

BASELINE_PREDICTORS = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXWT",
    "DBQ197",
    "DR1TVD",
]

DEFERRED_FEATURES = [
    "BMXHT",
    "BMXBMI",
    "DS1TVD",
    "DS1TCALC",
    "DBQ223A",
    "DBQ223B",
    "DBQ223C",
    "DBQ223D",
    "DBQ223E",
    "DBQ223U",
]

KNOWN_TARGET_DERIVED_VARIABLES = [
    "LBXVD2MS",
    "LBXVD3MS",
    "LBXVE3MS",
]

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

REGRESSION_METRICS = [
    "MAE",
    "RMSE",
    "R2",
]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.3 - MISSING DATA & PREPROCESSING")
    print("SUB-PHASE 3.3.6 - FINAL BASELINE DATASET SPECIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # -------------------------------------------------------------------------
    # EXPECTED DATASET SIZE
    # -------------------------------------------------------------------------

    assert len(df) == 702, (
        f"Expected 702 rows, found {len(df)}."
    )

    print("PASS: Analytical cohort contains exactly 702 rows.")

    # -------------------------------------------------------------------------
    # IDENTIFIER VALIDATION
    # -------------------------------------------------------------------------

    assert IDENTIFIER in df.columns
    assert df[IDENTIFIER].is_unique
    assert df[IDENTIFIER].notna().all()

    print(
        f"PASS: {IDENTIFIER} exists, is unique, "
        "and contains no missing values."
    )

    # -------------------------------------------------------------------------
    # TARGET VALIDATION
    # -------------------------------------------------------------------------

    assert TARGET in df.columns
    assert df[TARGET].notna().all()

    print(
        f"PASS: {TARGET} exists and contains no missing values."
    )

    # -------------------------------------------------------------------------
    # PREDICTOR VALIDATION
    # -------------------------------------------------------------------------

    missing_predictors = [
        feature
        for feature in BASELINE_PREDICTORS
        if feature not in df.columns
    ]

    assert not missing_predictors, (
        f"Missing baseline predictors: {missing_predictors}"
    )

    print("PASS: All baseline predictors are present.")

    # -------------------------------------------------------------------------
    # TARGET / ID EXCLUSION
    # -------------------------------------------------------------------------

    assert TARGET not in BASELINE_PREDICTORS
    assert IDENTIFIER not in BASELINE_PREDICTORS

    print("PASS: Target and identifier are excluded from X.")

    # -------------------------------------------------------------------------
    # TARGET-DERIVED VARIABLE CHECK
    # -------------------------------------------------------------------------

    target_derived_present = [
        feature
        for feature in BASELINE_PREDICTORS
        if feature in KNOWN_TARGET_DERIVED_VARIABLES
    ]

    assert not target_derived_present, (
        f"Target-derived variables found in predictors: "
        f"{target_derived_present}"
    )

    print(
        "PASS: No known target-derived laboratory variables "
        "are included."
    )

    # -------------------------------------------------------------------------
    # DUPLICATE PREDICTOR CHECK
    # -------------------------------------------------------------------------

    assert len(BASELINE_PREDICTORS) == len(
        set(BASELINE_PREDICTORS)
    )

    print("PASS: No duplicate baseline predictors.")

    # -------------------------------------------------------------------------
    # AGE VALIDATION
    # -------------------------------------------------------------------------

    age_min = df["RIDAGEYR"].min()
    age_max = df["RIDAGEYR"].max()

    assert age_min >= 1
    assert age_max <= 6

    print(
        f"PASS: Age range is {age_min:.0f}–{age_max:.0f} years."
    )

    # -------------------------------------------------------------------------
    # TARGET TYPE
    # -------------------------------------------------------------------------

    assert pd.api.types.is_numeric_dtype(df[TARGET])

    print(
        "PASS: Target is numeric and will be treated as "
        "a continuous regression target."
    )

    # -------------------------------------------------------------------------
    # MISSINGNESS REPORT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BASELINE PREDICTOR MISSINGNESS")
    print("=" * 80)

    missingness = pd.DataFrame(
        {
            "Feature": BASELINE_PREDICTORS,
            "Missing_Count": [
                df[feature].isna().sum()
                for feature in BASELINE_PREDICTORS
            ],
            "Missing_Percentage": [
                df[feature].isna().mean() * 100
                for feature in BASELINE_PREDICTORS
            ],
        }
    )

    print(
        missingness.to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # FINAL SPECIFICATION
    # -------------------------------------------------------------------------

    specification = {
        "project": "BalAarogya",
        "module": "Vitamin-D Risk Assessment",

        "dataset": {
            "name": "NHANES 2017-2018",
            "analytical_cohort": "Children aged 1-6 with valid total 25(OH)D",
            "rows": int(len(df)),
        },

        "identifier": IDENTIFIER,

        "target": {
            "name": TARGET,
            "type": "continuous",
            "unit": "nmol/L",
            "clinical_threshold_defined": False,
        },

        "baseline_predictors": BASELINE_PREDICTORS,

        "deferred_features": DEFERRED_FEATURES,

        "excluded_target_derived_variables":
            KNOWN_TARGET_DERIVED_VARIABLES,

        "missing_data": {
            "BMXWT": {
                "missing_count": int(df["BMXWT"].isna().sum()),
                "candidate_strategy": "median_imputation",
            },
            "DR1TVD": {
                "missing_count": int(df["DR1TVD"].isna().sum()),
                "candidate_strategies": [
                    "median_imputation",
                    "median_imputation_plus_missingness_indicator",
                    "feature_exclusion_ablation",
                ],
            },
        },

        "outlier_policy": {
            "automatic_deletion": False,
            "automatic_winsorization": False,
            "principle": (
                "Retain biologically plausible observations; "
                "flag unusual observations for diagnostic review."
            ),
        },

        "validation": {
            "test_size": TEST_SIZE,
            "cross_validation_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        },

        "initial_model_type": "regression",

        "initial_metrics": REGRESSION_METRICS,

        "leakage_control": {
            "preprocessing_fit_on_training_data_only": True,
            "feature_selection_training_only": True,
            "hyperparameter_tuning_training_only": True,
            "identifier_used_as_predictor": False,
            "target_used_as_predictor": False,
        },

        "clinical_interpretation": {
            "screening_threshold_defined": False,
            "diagnostic_claim": False,
            "blood_test_replacement_claim": False,
        },
    }

    # -------------------------------------------------------------------------
    # SAVE SPECIFICATION
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            specification,
            file,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("BASELINE MODEL SPECIFICATION")
    print("=" * 80)

    print(json.dumps(specification, indent=4))

    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Saved specification to:")
    print(OUTPUT_PATH)

    # -------------------------------------------------------------------------
    # FINAL VALIDATION
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    assert OUTPUT_PATH.exists()

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        saved_specification = json.load(file)

    assert saved_specification["dataset"]["rows"] == 702
    assert saved_specification["target"]["name"] == TARGET
    assert (
        saved_specification["baseline_predictors"]
        == BASELINE_PREDICTORS
    )
    assert saved_specification["identifier"] == IDENTIFIER

    print("PASS: Specification file successfully written.")
    print("PASS: Specification file successfully reloaded.")
    print("PASS: Core configuration matches frozen baseline.")

    print("\n" + "=" * 80)
    print("SUB-PHASE 3.3.6 COMPLETED")
    print("=" * 80)

    print(
        "The first baseline modeling specification has been frozen."
    )

    print(
        "\nNext phase: PHASE 4 - TARGET DEFINITION & "
        "CLINICAL LABEL STRATEGY."
    )


if __name__ == "__main__":
    main()