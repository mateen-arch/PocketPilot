"""PocketPilot Submission Packaging Script.

Packages project deliverables into a clean distribution zip archive
ready for submission or deployment.

Enforces:
- Single top-level directory: PocketPilot/
- Pre-packaging audit of mandatory deliverables (with warnings for pending presentation assets)
- Strict exclusion of environments (.venv, .venv_check), caches (__pycache__, .pytest_cache),
  git history (.git), model candidates, and distribution artifacts.
- Post-packaging verification re-opening the zip archive.
"""

import argparse
import os
from pathlib import Path
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent


def check_deliverables(root: Path, verbose: bool = True) -> dict[str, bool]:
    """Check existence of mandatory and presentation deliverables."""
    status = {}

    # 1. Source code
    has_streamlit = (root / "streamlit_app.py").is_file()
    has_src = (root / "src" / "pocketpilot").is_dir()
    status["source_code"] = has_streamlit and has_src

    # 2. Selected models and metadata
    models_dir = root / "models"
    has_battery = (models_dir / "battery_model.joblib").is_file()
    has_data = (models_dir / "data_usage_model.joblib").is_file()
    has_screen = (models_dir / "screen_time_model.joblib").is_file()
    has_meta = (models_dir / "metadata.json").is_file()
    status["selected_models_and_metadata"] = has_battery and has_data and has_screen and has_meta

    # 3. Evaluation evidence
    metrics_dir = root / "reports" / "metrics"
    has_metrics = metrics_dir.is_dir() and len(list(metrics_dir.glob("*.csv"))) > 0
    test_ev_dir = root / "reports" / "testing_evidence"
    has_test_ev = test_ev_dir.is_dir() and (test_ev_dir / "pytest_full_run.txt").is_file()
    status["evaluation_evidence"] = has_metrics and has_test_ev

    # 4. Project report (pdf or docx)
    report_dir = root / "docs" / "report"
    has_report = report_dir.is_dir() and any(
        f.suffix.lower() in (".pdf", ".docx") for f in report_dir.iterdir()
    ) if report_dir.is_dir() else False
    status["project_report"] = has_report

    # 5. Presentation slides (pdf or pptx)
    slides_dir = root / "docs" / "slides"
    has_slides = slides_dir.is_dir() and any(
        f.suffix.lower() in (".pdf", ".pptx") for f in slides_dir.iterdir()
    ) if slides_dir.is_dir() else False
    status["presentation_slides"] = has_slides

    # 6. Pipeline diagram (png, svg, or pdf)
    diagrams_dir = root / "docs" / "diagrams"
    has_diagram = diagrams_dir.is_dir() and any(
        f.suffix.lower() in (".png", ".svg", ".pdf") for f in diagrams_dir.iterdir()
    ) if diagrams_dir.is_dir() else False
    status["pipeline_diagram"] = has_diagram

    if verbose:
        print("=" * 70)
        print("DELIVERABLES AUDIT TABLE")
        print("=" * 70)
        deliverable_labels = {
            "source_code": "Source Code (streamlit_app.py & src/pocketpilot)",
            "selected_models_and_metadata": "Model Artifacts (models/*.joblib & metadata.json)",
            "evaluation_evidence": "Evaluation Evidence (reports/metrics & testing_evidence)",
            "project_report": "Project Report (docs/report/*.pdf or *.docx)",
            "presentation_slides": "Presentation Slides (docs/slides/*.pdf or *.pptx)",
            "pipeline_diagram": "Pipeline Architecture Diagram (docs/diagrams/*.png/svg/pdf)",
        }
        for k, label in deliverable_labels.items():
            is_present = status[k]
            state_str = "PRESENT" if is_present else "MISSING"
            print(f"  {label:<55}: [{state_str}]")

        # Emit clear warnings for pending external presentation assets
        for k, label in deliverable_labels.items():
            if not status[k]:
                if k in ("project_report", "presentation_slides", "pipeline_diagram"):
                    print(f"  WARNING: Deliverable '{label}' is MISSING. Continuing packaging as permitted.")
                else:
                    print(f"  CRITICAL WARNING: Mandatory deliverable '{label}' is MISSING.")
        print("=" * 70)

    return status


def is_excluded(rel_path: str) -> bool:
    """Check if relative file path matches any packaging exclusion rule."""
    parts = Path(rel_path).parts
    filename = Path(rel_path).name

    # Check for excluded directory components anywhere in the path
    for p in parts:
        if p.startswith(".venv") or p in (
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ipynb_checkpoints",
            "dist",
            "node_modules",
            "candidates",
        ):
            return True

    # Check for excluded file extensions or names
    if filename.endswith(".pyc") or filename == ".DS_Store" or filename == "Thumbs.db":
        return True

    return False


def build_zip(
    root: Path,
    output_path: Path,
    verbose: bool = True,
) -> tuple[int, float, list[tuple[str, float]]]:
    """
    Build a clean distribution zip archive with top-level folder 'PocketPilot'.

    Parameters
    ----------
    root : Path
        Project root directory.
    output_path : Path
        Destination zip path.
    verbose : bool, default=True
        Whether to print audit and summary information.

    Returns
    -------
    tuple[int, float, list[tuple[str, float]]]
        (total_files_count, total_size_mb, top_10_largest_files)
    """
    check_deliverables(root, verbose=verbose)

    # Top-level files and directories to include
    include_targets = [
        "README.md",
        "requirements.txt",
        "streamlit_app.py",
        ".streamlit",
        "app",
        "src",
        "scripts",
        "tests",
        "models",
        "data",
        "reports",
        "docs",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    files_to_pack: list[tuple[Path, str]] = []

    for target_name in include_targets:
        target_path = root / target_name
        if not target_path.exists():
            continue

        if target_path.is_file():
            rel_str = target_name
            if not is_excluded(rel_str):
                files_to_pack.append((target_path, rel_str))
        elif target_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(target_path):
                # Filter out excluded directories in-place to avoid descending into them
                dirnames[:] = [d for d in dirnames if not is_excluded(d)]
                for fname in filenames:
                    fpath = Path(dirpath) / fname
                    rel_to_root = str(fpath.relative_to(root)).replace("\\", "/")
                    if not is_excluded(rel_to_root):
                        files_to_pack.append((fpath, rel_to_root))

    if verbose:
        print(f"\nCompressing {len(files_to_pack)} files into {output_path}...")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for fpath, rel_str in files_to_pack:
            # Single top-level directory named PocketPilot
            arcname = f"PocketPilot/{rel_str}"
            zip_out.write(fpath, arcname=arcname)

    # Verify created zip
    total_files = 0
    file_sizes: list[tuple[str, float]] = []

    with zipfile.ZipFile(output_path, "r") as zip_in:
        names = zip_in.namelist()
        total_files = len(names)

        # Check required files
        required_in_zip = [
            "PocketPilot/streamlit_app.py",
            "PocketPilot/requirements.txt",
            "PocketPilot/models/battery_model.joblib",
        ]
        for req in required_in_zip:
            if req not in names:
                raise RuntimeError(f"Zip verification failed: missing required file '{req}'")

        # Check excluded patterns
        for name in names:
            name_lower = name.lower()
            if any(exc in name_lower for exc in (".venv", "__pycache__", "candidates", ".git/")):
                raise RuntimeError(f"Zip verification failed: found forbidden artifact '{name}' in archive")

            info = zip_in.getinfo(name)
            file_sizes.append((name, info.file_size / (1024 * 1024)))

    file_sizes.sort(key=lambda x: x[1], reverse=True)
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)

    if verbose:
        print("\n" + "=" * 70)
        print("SUBMISSION ZIP SUMMARY")
        print("=" * 70)
        print(f"  Archive Path:    {output_path.resolve()}")
        print(f"  Archive Size:    {zip_size_mb:.2f} MB")
        print(f"  Total Files:     {total_files}")
        print("\n  Top 10 Largest Files in Archive:")
        for name, sz in file_sizes[:10]:
            print(f"    * {name} ({sz:.2f} MB)")
        print("=" * 70)
        print("  Archive verification PASSED successfully.")

    return total_files, zip_size_mb, file_sizes[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Package PocketPilot for final submission.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "dist" / "PocketPilot_submission.zip"),
        help="Destination path for submission zip archive.",
    )
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    build_zip(root=ROOT_DIR, output_path=out_path, verbose=True)


if __name__ == "__main__":
    main()
