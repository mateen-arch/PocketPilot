"""
Dataset splitting module for PocketPilot.

Architecture and Leakage Prevention Note:
The train/test split happens strictly BEFORE any label derivation, scaling, or encoding.
The test set represents unseen future deployment data and is held out exclusively for final
model evaluation and reporting. Any cross-validation, hyperparameter tuning, feature scaling,
and categorical encoding fit must be conducted solely inside the training split to guarantee
zero data leakage.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from pocketpilot.config import (
    PROCESSED_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RANDOM_SEED,
    TEST_SIZE,
)


def split_ds1(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Dataset 1 into train and test partitions stratified by behavior_class.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Dataset 1 DataFrame.
    test_size : float, default=TEST_SIZE (0.2)
        Fraction of data reserved for the test split.
    seed : int, default=RANDOM_SEED (42)
        Random state seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (train_df, test_df), each returned as independent copies with reset indices.

    Raises
    ------
    AssertionError
        If there is any user_id overlap between train and test sets, or if
        the sum of split row counts does not match the input length.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["behavior_class"],
    )

    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # Integrity assertions to ensure absolute partition independence
    assert set(train_df["user_id"]).isdisjoint(
        set(test_df["user_id"])
    ), "Data leakage detected: user_id overlap between train and test splits!"

    assert (
        len(train_df) + len(test_df) == len(df)
    ), f"Split row count mismatch: {len(train_df)} train + {len(test_df)} test != {len(df)} total"

    return train_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path | str = PROCESSED_TRAIN_PATH,
    test_path: Path | str = PROCESSED_TEST_PATH,
) -> tuple[Path, Path]:
    """
    Persist train and test splits to CSV files.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    test_df : pd.DataFrame
        Testing partition.
    train_path : Path or str, default=PROCESSED_TRAIN_PATH
        Destination path for training CSV.
    test_path : Path or str, default=PROCESSED_TEST_PATH
        Destination path for testing CSV.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved (train_path, test_path) files.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path
