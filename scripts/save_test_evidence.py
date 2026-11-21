"""
Script to collect and save pytest testing evidence and test inventory.
"""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "reports" / "testing_evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def get_environment_header() -> str:
    """Build standardized header with timestamp and environment package versions."""
    import numpy as np
    import pandas as pd
    import sklearn
    import streamlit as st
    import xgboost as xgb

    now_iso = datetime.now(timezone.utc).isoformat()
    header_lines = [
        "=" * 70,
        f"POCKETPILOT TEST EXECUTION EVIDENCE",
        f"Timestamp (UTC): {now_iso}",
        f"Python: {sys.version.split()[0]} ({sys.executable})",
        f"scikit-learn: {sklearn.__version__}",
        f"xgboost: {xgb.__version__}",
        f"pandas: {pd.__version__}",
        f"numpy: {np.__version__}",
        f"streamlit: {st.__version__}",
        "=" * 70,
        "",
    ]
    return "\n".join(header_lines)


def main() -> None:
    print("=" * 70)
    print("SAVING TEST EVIDENCE & INVENTORY")
    print("=" * 70)

    header = get_environment_header()

    # 1. Run pytest full execution with -rA (report all details)
    full_cmd = [sys.executable, "-m", "pytest", "-q", "-rA"]
    res_full = subprocess.run(full_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    full_output = header + res_full.stdout + (("\nSTDERR:\n" + res_full.stderr) if res_full.stderr else "")

    full_file = EVIDENCE_DIR / "pytest_full_run.txt"
    full_file.write_text(full_output, encoding="utf-8")
    print(f"Saved pytest run output to: {full_file.relative_to(PROJECT_ROOT)}")

    # 2. Run pytest collection only
    collect_cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    res_collect = subprocess.run(collect_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    collect_output = header + res_collect.stdout + (("\nSTDERR:\n" + res_collect.stderr) if res_collect.stderr else "")

    inventory_file = EVIDENCE_DIR / "test_inventory.txt"
    inventory_file.write_text(collect_output, encoding="utf-8")
    print(f"Saved test inventory to: {inventory_file.relative_to(PROJECT_ROOT)}")

    # 3. Parse and print test counts per test file
    print("\n--- TEST COUNTS PER TEST FILE ---")
    test_counts = {}
    for line in res_collect.stdout.splitlines():
        line = line.strip()
        if "::" in line:
            test_file = line.split("::")[0].strip()
            test_counts[test_file] = test_counts.get(test_file, 0) + 1

    total_tests = sum(test_counts.values())
    for f_name, count in sorted(test_counts.items()):
        print(f"  * {f_name}: {count} tests")
    print(f"Total collected tests: {total_tests}")


if __name__ == "__main__":
    main()
