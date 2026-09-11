from pathlib import Path

import pandas as pd


# ============================================================
# BalAarogya - Vitamin-D Module
# NHANES Merge Integrity Check
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "VID": "VID_J.XPT",
    "DEMO": "DEMO_J.XPT",
    "BMX": "BMX_J.XPT",
    "DBQ": "DBQ_J.XPT",
    "DR1TOT": "DR1TOT_J.XPT",
    "DS1TOT": "DS1TOT_J.XPT",
}


def load_seqn(file_name: str) -> pd.DataFrame:
    """Load SEQN from an NHANES XPT file."""
    file_path = RAW_DATA_DIR / file_name

    df = pd.read_sas(
        file_path,
        format="xport",
    )

    return df[["SEQN"]]

def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("NHANES 2017-2018 MERGE INTEGRITY CHECK")
    print("=" * 80)

    datasets = {}

    # --------------------------------------------------------
    # 1. Load SEQN from every dataset
    # --------------------------------------------------------

    print("\n--- DATASET SIZES & SEQN CHECK ---")

    for name, file_name in DATASETS.items():

        df = load_seqn(file_name)

        datasets[name] = df

        duplicate_count = df["SEQN"].duplicated().sum()
        missing_count = df["SEQN"].isna().sum()
        unique_count = df["SEQN"].nunique()

        print(
            f"{name:8s} | "
            f"Rows: {len(df):5,d} | "
            f"Unique SEQN: {unique_count:5,d} | "
            f"Duplicates: {duplicate_count:3,d} | "
            f"Missing SEQN: {missing_count:3,d}"
        )

    # --------------------------------------------------------
    # 2. Compare VID against every other dataset
    # --------------------------------------------------------

    print("\n--- VID OVERLAP WITH OTHER DATASETS ---")

    vid_seqn = set(datasets["VID"]["SEQN"])

    for name, df in datasets.items():

        if name == "VID":
            continue

        other_seqn = set(df["SEQN"])

        overlap = len(vid_seqn & other_seqn)
        vid_only = len(vid_seqn - other_seqn)
        other_only = len(other_seqn - vid_seqn)

        print(
            f"VID ↔ {name:6s} | "
            f"Overlap: {overlap:5,d} | "
            f"VID only: {vid_only:5,d} | "
            f"{name} only: {other_only:5,d}"
        )

    # --------------------------------------------------------
    # 3. Construct intersection across all six datasets
    # --------------------------------------------------------

    print("\n--- COMMON PARTICIPANTS ACROSS ALL SIX DATASETS ---")

    common_seqn = None

    for df in datasets.values():

        current_seqn = set(df["SEQN"])

        if common_seqn is None:
            common_seqn = current_seqn
        else:
            common_seqn &= current_seqn

    print(
        f"Participants present in ALL SIX datasets: "
        f"{len(common_seqn):,}"
    )

    # --------------------------------------------------------
    # 4. Restrict common participants to age 1-6
    # --------------------------------------------------------

    print("\n--- CHILD COHORT CHECK ---")

    demo = pd.read_sas(
    RAW_DATA_DIR / DATASETS["DEMO"],
    format="xport",
    )[["SEQN", "RIDAGEYR"]]

    demo_children = demo[
        demo["RIDAGEYR"].between(1, 6, inclusive="both")
    ]

    child_seqn = set(demo_children["SEQN"])

    common_children = common_seqn & child_seqn

    print(
        f"Children aged 1-6 present in all six datasets: "
        f"{len(common_children):,}"
    )

    # --------------------------------------------------------
    # 5. Check valid Vitamin-D measurements
    # --------------------------------------------------------

    print("\n--- VALID VITAMIN-D GROUND TRUTH ---")

    vid = pd.read_sas(
    RAW_DATA_DIR / DATASETS["VID"],
    format="xport",
    )[["SEQN", "LBXVIDMS"]]

    valid_vitd = vid[vid["LBXVIDMS"].notna()]

    valid_vitd_seqn = set(valid_vitd["SEQN"])

    valid_children_vitd = valid_vitd_seqn & child_seqn
    valid_children_all_six = valid_children_vitd & common_seqn

    print(
        f"Valid 25(OH)D measurements (all ages): "
        f"{len(valid_vitd_seqn):,}"
    )

    print(
        f"Children aged 1-6 with valid 25(OH)D: "
        f"{len(valid_children_vitd):,}"
    )

    print(
        f"Children aged 1-6 with valid 25(OH)D "
        f"and present in all six datasets: "
        f"{len(valid_children_all_six):,}"
    )

    print("\n" + "=" * 80)
    print("MERGE INTEGRITY CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()