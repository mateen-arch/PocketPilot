"""Tests for submission packaging script and exclusion guards.

Verifies:
- All packaged files reside within a single top-level 'PocketPilot/' directory.
- Required production files (streamlit_app.py, requirements.txt, models) are included.
- Excluded paths (.venv, __pycache__, candidates, .git) are strictly omitted.
"""

from pathlib import Path
import zipfile
import pytest

from scripts.package_submission import build_zip


def test_build_zip_exclusions_and_structure(tmp_path):
    """Verify build_zip includes valid project files and enforces strict artifact exclusions."""
    fake_root = tmp_path / "project_root"
    fake_root.mkdir()

    # Create expected files
    (fake_root / "streamlit_app.py").write_text("# app entry", encoding="utf-8")
    (fake_root / "requirements.txt").write_text("pandas\nnumpy", encoding="utf-8")
    (fake_root / "README.md").write_text("# Readme", encoding="utf-8")

    src_dir = fake_root / "src" / "pocketpilot"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("", encoding="utf-8")

    models_dir = fake_root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "battery_model.joblib").write_bytes(b"model_bytes")
    (models_dir / "data_usage_model.joblib").write_bytes(b"data_bytes")
    (models_dir / "screen_time_model.joblib").write_bytes(b"screen_bytes")
    (models_dir / "metadata.json").write_text("{}", encoding="utf-8")

    # Create files that MUST be excluded
    candidates_dir = models_dir / "candidates"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "candidate_model.joblib").write_bytes(b"candidate")

    venv_dir = fake_root / ".venv_check" / "Scripts"
    venv_dir.mkdir(parents=True)
    (venv_dir / "python.exe").write_bytes(b"fake_python")

    pycache_dir = src_dir / "__pycache__"
    pycache_dir.mkdir(parents=True)
    (pycache_dir / "test.cpython-312.pyc").write_bytes(b"pyc")

    git_dir = fake_root / ".git" / "objects"
    git_dir.mkdir(parents=True)
    (git_dir / "commit_obj").write_bytes(b"git")

    zip_out = tmp_path / "output" / "test_submission.zip"

    total_files, size_mb, top_files = build_zip(
        root=fake_root,
        output_path=zip_out,
        verbose=False,
    )

    assert zip_out.is_file()
    assert total_files > 0
    assert size_mb >= 0.0

    with zipfile.ZipFile(zip_out, "r") as zf:
        names = zf.namelist()

        # 1. Single top-level directory named PocketPilot
        assert all(n.startswith("PocketPilot/") for n in names)

        # 2. Contains required files
        assert "PocketPilot/streamlit_app.py" in names
        assert "PocketPilot/requirements.txt" in names
        assert "PocketPilot/models/battery_model.joblib" in names
        assert "PocketPilot/src/pocketpilot/__init__.py" in names

        # 3. Strictly excludes forbidden folders/artifacts
        for n in names:
            n_lower = n.lower()
            assert ".venv" not in n_lower, f"Found .venv file in archive: {n}"
            assert "__pycache__" not in n_lower, f"Found __pycache__ file in archive: {n}"
            assert "candidates" not in n_lower, f"Found candidate file in archive: {n}"
            assert ".git/" not in n_lower, f"Found .git file in archive: {n}"
            assert not n_lower.endswith(".pyc"), f"Found .pyc file in archive: {n}"
