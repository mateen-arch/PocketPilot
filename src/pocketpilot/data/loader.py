"""
Data loader module for primary and extension mobile usage datasets.
"""

from pathlib import Path
import pandas as pd

from pocketpilot.config import (
    COLUMN_RENAME,
    DS1_PATH,
    DS2_PATH,
    DS2_READ_KWARGS,
)


def load_ds1(path: Path | str = DS1_PATH) -> pd.DataFrame:
    """
    Load Dataset 1 (Mobile Device Usage and User Behavior Dataset) from CSV.

    Validates that all expected raw columns are present before returning the unmutated DataFrame.

    Parameters
    ----------
    path : Path or str, default=DS1_PATH
        Path to raw Dataset 1 CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with original column names and dtypes.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If any of the 11 required raw columns is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset 1 file not found at {path}")

    df = pd.read_csv(path)
    expected_columns = set(COLUMN_RENAME.keys())
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset 1 is missing required column(s): {sorted(missing_columns)}"
        )
    return df


def load_ds2(path: Path | str = DS2_PATH) -> pd.DataFrame:
    """
    Load Dataset 2 (Mobile App Usage & Screen Time 2022-2024) from CSV.

    Uses DS2_READ_KWARGS (keep_default_na=False) so that literal 'None' entries
    in sleep_disruption_from_phone are preserved as strings rather than NaN.

    Parameters
    ----------
    path : Path or str, default=DS2_PATH
        Path to raw Dataset 2 CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset preserving literal category strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If the loaded DataFrame is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset 2 file not found at {path}")

    df = pd.read_csv(path, **DS2_READ_KWARGS)
    if df.empty:
        raise ValueError(f"Dataset 2 at {path} is empty.")
    return df
