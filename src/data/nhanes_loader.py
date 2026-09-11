from pathlib import Path

import pandas as pd


# ============================================================
# BalAarogya - Vitamin-D Module
# Centralized NHANES XPT Loader
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# pandas' XPORT reader can decode an IBM floating-point
# representation of zero as this tiny number.
SAS_XPORT_ZERO_ARTIFACT = 5.397605346934028e-79


def load_xpt(
    file_name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load an NHANES XPT file and normalize the XPORT zero artifact.

    Parameters
    ----------
    file_name:
        Name of the XPT file inside data/raw/.

    columns:
        Optional list of columns to retain after loading.

    Returns
    -------
    pandas.DataFrame
        Loaded and minimally normalized NHANES dataset.
    """

    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"NHANES file not found: {file_path}"
        )

    # --------------------------------------------------------
    # Load XPT
    # --------------------------------------------------------

    df = pd.read_sas(
        file_path,
        format="xport",
    )

    # --------------------------------------------------------
    # Normalize pandas XPORT zero artifact
    # --------------------------------------------------------

    df = df.replace(
        SAS_XPORT_ZERO_ARTIFACT,
        0.0,
    )

    # --------------------------------------------------------
    # Select requested columns
    # --------------------------------------------------------

    if columns is not None:

        missing_columns = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Columns not found in {file_name}: "
                f"{missing_columns}"
            )

        df = df[columns].copy()

    return df