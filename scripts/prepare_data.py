"""
Preparation script for Dataset 1.

Executes data loading, cleaning, validation, stratified splitting, and persistence.
Can be executed from the project root using:
    python scripts/prepare_data.py
"""

from pathlib import Path
import sys

# Ensure src directory is on sys.path when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pocketpilot.data.cleaning import clean_ds1
from pocketpilot.data.loader import load_ds1
from pocketpilot.data.splitting import save_splits, split_ds1


def main() -> None:
    print("=" * 60)
    print("PocketPilot Data Preparation Pipeline")
    print("=" * 60)

    # 1. Load raw dataset
    print("[1/4] Loading raw Dataset 1...")
    raw_df = load_ds1()
    print(f"      Loaded {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    # 2. Clean and validate dataset
    print("[2/4] Cleaning and validating data ranges...")
    clean_df = clean_ds1(raw_df)
    print(f"      Cleaned dataset contains {len(clean_df)} validated rows.")

    # 3. Stratified train/test split
    print("[3/4] Performing stratified train/test split...")
    train_df, test_df = split_ds1(clean_df)
    print(f"      Train split: {len(train_df)} rows")
    print(f"      Test split:  {len(test_df)} rows")

    print("\nBehavior Class Proportions:")
    train_props = train_df["behavior_class"].value_counts(normalize=True).sort_index()
    test_props = test_df["behavior_class"].value_counts(normalize=True).sort_index()
    for cls in sorted(clean_df["behavior_class"].unique()):
        print(
            f"  Class {cls}: Train = {train_props.get(cls, 0.0):.2%}, "
            f"Test = {test_props.get(cls, 0.0):.2%}"
        )

    # 4. Save processed partitions
    print("\n[4/4] Saving split datasets to disk...")
    train_path, test_path = save_splits(train_df, test_df)
    print(f"      Saved training data: {train_path}")
    print(f"      Saved test data:     {test_path}")

    print("=" * 60)
    print("Data preparation complete successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
