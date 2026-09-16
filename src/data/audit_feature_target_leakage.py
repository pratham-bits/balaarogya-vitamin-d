from pathlib import Path
import pandas as pd


# =============================================================================
# BalAarogya - Vitamin-D Module
# PHASE 3.3 - MISSING DATA & PREPROCESSING
# SUB-PHASE 3.3.5 - TARGET / FEATURE SPLIT & LEAKAGE AUDIT
# =============================================================================


# -----------------------------------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vitamin_d_nhanes_2017_2018.csv"
)


# -----------------------------------------------------------------------------
# MODELING SPECIFICATION
# -----------------------------------------------------------------------------

TARGET = "LBXVIDMS"

IDENTIFIER = "SEQN"

BASELINE_PREDICTORS = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXWT",
    "DBQ197",
    "DR1TVD",
]

# Optional missingness indicator to be used in a later controlled experiment.
# It is NOT present in the current analytical dataset.
OPTIONAL_MISSINGNESS_INDICATORS = [
    "DR1TVD_missing",
    "BMXWT_missing",
]

# Known variables from the original NHANES Vitamin D laboratory component
# that must never be used as predictors because they are directly related
# to the laboratory target.
KNOWN_TARGET_DERIVED_VARIABLES = [
    "LBXVD2MS",
    "LBXVD3MS",
    "LBXVE3MS",
]


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_pass(condition: bool, message: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {message}")
    return condition


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.3 - MISSING DATA & PREPROCESSING")
    print("SUB-PHASE 3.3.5 - TARGET / FEATURE SPLIT & LEAKAGE AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------

    print_section("DATASET")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset path: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nDataset columns:")
    for column in df.columns:
        print(f"  - {column}")

    # -------------------------------------------------------------------------
    # DECLARED MODELING ROLES
    # -------------------------------------------------------------------------

    print_section("DECLARED MODELING ROLES")

    print(f"Target:")
    print(f"  {TARGET}")

    print("\nIdentifier:")
    print(f"  {IDENTIFIER}")

    print("\nBaseline predictors:")
    for feature in BASELINE_PREDICTORS:
        print(f"  - {feature}")

    print("\nOptional missingness indicators for later experiments:")
    for feature in OPTIONAL_MISSINGNESS_INDICATORS:
        print(f"  - {feature}")

    # -------------------------------------------------------------------------
    # AUDIT 1 - TARGET EXISTS
    # -------------------------------------------------------------------------

    print_section("AUDIT 1 - TARGET VALIDATION")

    target_exists = check_pass(
        TARGET in df.columns,
        f"Target column '{TARGET}' exists in the analytical dataset."
    )

    if not target_exists:
        raise RuntimeError(
            f"Target column '{TARGET}' was not found."
        )

    target_missing = df[TARGET].isna().sum()

    target_complete = check_pass(
        target_missing == 0,
        f"Target completeness: {target_missing} missing values."
    )

    # -------------------------------------------------------------------------
    # AUDIT 2 - PREDICTOR EXISTENCE
    # -------------------------------------------------------------------------

    print_section("AUDIT 2 - PREDICTOR COLUMN VALIDATION")

    missing_predictors = [
        feature
        for feature in BASELINE_PREDICTORS
        if feature not in df.columns
    ]

    predictors_exist = check_pass(
        len(missing_predictors) == 0,
        "All declared baseline predictors exist."
    )

    if missing_predictors:
        print("\nMissing predictors:")
        for feature in missing_predictors:
            print(f"  - {feature}")

    # -------------------------------------------------------------------------
    # AUDIT 3 - TARGET NOT IN PREDICTORS
    # -------------------------------------------------------------------------

    print_section("AUDIT 3 - DIRECT TARGET LEAKAGE")

    target_in_predictors = TARGET in BASELINE_PREDICTORS

    direct_target_leakage = check_pass(
        not target_in_predictors,
        f"Target '{TARGET}' is not included in the predictor list."
    )

    # -------------------------------------------------------------------------
    # AUDIT 4 - IDENTIFIER EXCLUSION
    # -------------------------------------------------------------------------

    print_section("AUDIT 4 - IDENTIFIER EXCLUSION")

    identifier_in_predictors = IDENTIFIER in BASELINE_PREDICTORS

    identifier_excluded = check_pass(
        not identifier_in_predictors,
        f"Identifier '{IDENTIFIER}' is excluded from model predictors."
    )

    # -------------------------------------------------------------------------
    # AUDIT 5 - DUPLICATE PREDICTOR NAMES
    # -------------------------------------------------------------------------

    print_section("AUDIT 5 - PREDICTOR NAME DUPLICATION")

    duplicate_predictors = (
        pd.Series(BASELINE_PREDICTORS)
        .duplicated()
    )

    duplicate_predictor_names = (
        pd.Series(BASELINE_PREDICTORS)[duplicate_predictors]
        .tolist()
    )

    no_duplicate_predictors = check_pass(
        len(duplicate_predictor_names) == 0,
        "No duplicate predictor names."
    )

    if duplicate_predictor_names:
        print("\nDuplicate predictors:")
        for feature in duplicate_predictor_names:
            print(f"  - {feature}")

    # -------------------------------------------------------------------------
    # AUDIT 6 - KNOWN TARGET-DERIVED VARIABLES
    # -------------------------------------------------------------------------

    print_section("AUDIT 6 - KNOWN TARGET-DERIVED VARIABLES")

    target_derived_in_predictors = [
        feature
        for feature in BASELINE_PREDICTORS
        if feature in KNOWN_TARGET_DERIVED_VARIABLES
    ]

    no_target_derived_predictors = check_pass(
        len(target_derived_in_predictors) == 0,
        "No known target-derived Vitamin-D laboratory variables are used."
    )

    if target_derived_in_predictors:
        print("\nTarget-derived variables incorrectly included:")
        for feature in target_derived_in_predictors:
            print(f"  - {feature}")

    print("\nKnown excluded target-derived variables:")
    for feature in KNOWN_TARGET_DERIVED_VARIABLES:
        print(f"  - {feature}")

    # -------------------------------------------------------------------------
    # AUDIT 7 - OPTIONAL MISSINGNESS INDICATORS
    # -------------------------------------------------------------------------

    print_section("AUDIT 7 - MISSINGNESS INDICATOR CHECK")

    print(
        "Missingness indicators are derived only from predictor availability "
        "and do not use the laboratory target."
    )

    missingness_indicator_check = True

    for indicator in OPTIONAL_MISSINGNESS_INDICATORS:

        if indicator not in df.columns:
            print(
                f"INFO: '{indicator}' is not currently present in the "
                "analytical dataset. It will be generated inside the "
                "future preprocessing pipeline."
            )
        else:
            print(
                f"INFO: '{indicator}' already exists in the dataset."
            )

    print(
        "\nPASS: Missingness indicators are structurally independent "
        "of the target."
    )

    # -------------------------------------------------------------------------
    # AUDIT 8 - TARGET IS NOT USED AS PREDICTOR THROUGH COLUMN OVERLAP
    # -------------------------------------------------------------------------

    print_section("AUDIT 8 - X / y COLUMN SEPARATION")

    predictor_target_overlap = set(BASELINE_PREDICTORS).intersection(
        {TARGET}
    )

    x_y_separated = check_pass(
        len(predictor_target_overlap) == 0,
        "Predictor matrix X and target y have no overlapping columns."
    )

    # -------------------------------------------------------------------------
    # AUDIT 9 - IDENTIFIER / TARGET SEPARATION
    # -------------------------------------------------------------------------

    print_section("AUDIT 9 - IDENTIFIER / TARGET SEPARATION")

    id_target_overlap = set(BASELINE_PREDICTORS).intersection(
        {IDENTIFIER, TARGET}
    )

    clean_feature_roles = check_pass(
        len(id_target_overlap) == 0,
        "Predictors contain neither the identifier nor the target."
    )

    # -------------------------------------------------------------------------
    # FINAL FEATURE TABLE
    # -------------------------------------------------------------------------

    print_section("FINAL BASELINE FEATURE / TARGET SPECIFICATION")

    print("X (predictors):")

    for i, feature in enumerate(BASELINE_PREDICTORS, start=1):
        print(f"  {i}. {feature}")

    print("\ny (target):")
    print(f"  {TARGET}")

    print("\nIdentifier:")
    print(f"  {IDENTIFIER}")

    # -------------------------------------------------------------------------
    # DATASET-LEVEL SANITY CHECK
    # -------------------------------------------------------------------------

    print_section("DATASET-LEVEL SANITY CHECK")

    expected_columns = (
        [IDENTIFIER]
        + BASELINE_PREDICTORS
        + [TARGET]
    )

    unexpected_role_overlap = set(BASELINE_PREDICTORS).intersection(
        set([IDENTIFIER, TARGET])
    )

    sanity_pass = check_pass(
        len(unexpected_role_overlap) == 0,
        "No identifier/target overlap exists in the baseline predictor set."
    )

    print("\nRows available for baseline specification:", len(df))

    # -------------------------------------------------------------------------
    # FINAL AUDIT RESULT
    # -------------------------------------------------------------------------

    print_section("FINAL LEAKAGE AUDIT RESULT")

    all_checks = [
        target_exists,
        target_complete,
        predictors_exist,
        direct_target_leakage,
        identifier_excluded,
        no_duplicate_predictors,
        no_target_derived_predictors,
        missingness_indicator_check,
        x_y_separated,
        clean_feature_roles,
        sanity_pass,
    ]

    overall_pass = all(all_checks)

    if overall_pass:
        print("OVERALL LEAKAGE AUDIT: PASS")
    else:
        print("OVERALL LEAKAGE AUDIT: FAIL")

    # -------------------------------------------------------------------------
    # IMPORTANT METHODOLOGICAL NOTES
    # -------------------------------------------------------------------------

    print_section("METHODOLOGICAL NOTES")

    print(
        "1. This audit does not prove that every future modeling step "
        "will be leakage-free."
    )

    print(
        "2. Imputation, scaling, feature selection, dimensionality reduction, "
        "and hyperparameter tuning must be fitted using training data only."
    )

    print(
        "3. SEQN is retained only as an identifier and must not be supplied "
        "to the ML model."
    )

    print(
        "4. LBXVIDMS is the laboratory target and must not be used as an "
        "input feature."
    )

    print(
        "5. LBXVD2MS, LBXVD3MS, and LBXVE3MS are excluded because they are "
        "directly related to the Vitamin-D laboratory target."
    )

    print(
        "6. Outlier flags from Phase 3.3.3 do not change feature eligibility."
    )

    print(
        "7. This script performs no data modification."
    )

    print("\n" + "=" * 80)
    print("SUB-PHASE 3.3.5 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()