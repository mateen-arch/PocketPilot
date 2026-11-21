"""Fresh clone verification script for PocketPilot.

Proves that the clean, committed git tree contains all necessary assets to:
1. Pass the entire automated test suite (pytest -q).
2. Load production ML pipelines and run model inference on the typical user persona.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent.parent


def remove_readonly(func, path, exc_info):
    """Clear the read-only attribute on Windows git objects before directory removal."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def run_fresh_clone_check() -> None:
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)
    ).decode().strip()

    timestamp = datetime.now(timezone.utc).isoformat()
    python_ver = sys.version.replace("\n", " ")

    clone_dir = tempfile.mkdtemp(prefix="pocketpilot_clone_")
    output_log = []

    header = (
        "======================================================================\n"
        "POCKETPILOT FRESH CLONE VERIFICATION EVIDENCE\n"
        "======================================================================\n"
        f"Timestamp (UTC): {timestamp}\n"
        f"Python Version:  {python_ver}\n"
        f"Commit Tested:   {commit_hash}\n"
        f"Temporary Clone: {clone_dir}\n"
        "======================================================================\n\n"
    )
    print(header)
    output_log.append(header)

    try:
        print(f"Cloning local repository from {ROOT_DIR} into {clone_dir}...")
        clone_cmd = ["git", "clone", "--local", "--no-hardlinks", str(ROOT_DIR), clone_dir]
        subprocess.check_call(clone_cmd)
        output_log.append(f"[Git Clone] Successfully created clean local clone at {clone_dir}\n\n")

        print("Running pytest -q inside the fresh clone...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(clone_dir) / "src")

        pytest_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=clone_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        pytest_out = (
            "--- PYTEST EXECUTION IN FRESH CLONE ---\n"
            f"Exit Code: {pytest_proc.returncode}\n"
            "STDOUT:\n" + pytest_proc.stdout + "\n"
            "STDERR:\n" + pytest_proc.stderr + "\n"
        )
        print(pytest_out)
        output_log.append(pytest_out)

        if pytest_proc.returncode != 0:
            raise RuntimeError(
                f"Pytest failed inside fresh clone with code {pytest_proc.returncode}!\n"
                f"STDOUT:\n{pytest_proc.stdout}\nSTDERR:\n{pytest_proc.stderr}"
            )

        print("Running inference test on typical user persona inside fresh clone...")
        inference_code = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path('src').resolve()))\n"
            "from pocketpilot.inference.predictor import load_artifacts, forecast\n"
            "artifacts = load_artifacts()\n"
            "inputs = {\n"
            "    'device_model': 'Samsung Galaxy S21',\n"
            "    'app_usage_min': 250.0,\n"
            "    'screen_on_hours': 5.0,\n"
            "    'apps_installed': 50,\n"
            "}\n"
            "budgets = {'battery_mah': 1800.0, 'data_mb': 1000.0, 'screen_hours': 6.0}\n"
            "res = forecast(artifacts, inputs, budgets, screen_known=True)\n"
            "pred_bat = res['predictions']['battery_mah']\n"
            "print(f'Typical user predicted battery drain: {pred_bat:.2f} mAh')\n"
        )

        infer_proc = subprocess.run(
            [sys.executable, "-c", inference_code],
            cwd=clone_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        infer_out = (
            "--- INFERENCE CHECK IN FRESH CLONE ---\n"
            f"Exit Code: {infer_proc.returncode}\n"
            "STDOUT:\n" + infer_proc.stdout + "\n"
            "STDERR:\n" + infer_proc.stderr + "\n"
        )
        print(infer_out)
        output_log.append(infer_out)

        if infer_proc.returncode != 0:
            raise RuntimeError(
                f"Inference check failed inside fresh clone with code {infer_proc.returncode}!\n"
                f"STDOUT:\n{infer_proc.stdout}\nSTDERR:\n{infer_proc.stderr}"
            )

        output_log.append("\n[VERDICT] Fresh clone passed all tests and inference checks cleanly!\n")
        print("\n[VERDICT] Fresh clone passed all tests and inference checks cleanly!")

    finally:
        print(f"Cleaning up temporary clone directory {clone_dir}...")
        shutil.rmtree(clone_dir, onerror=remove_readonly)

    ev_dir = ROOT_DIR / "reports" / "testing_evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    ev_file = ev_dir / "fresh_clone_check.txt"
    ev_file.write_text("".join(output_log), encoding="utf-8")
    print(f"Saved fresh clone evidence -> {ev_file.relative_to(ROOT_DIR)}")


def main() -> None:
    run_fresh_clone_check()


if __name__ == "__main__":
    main()
