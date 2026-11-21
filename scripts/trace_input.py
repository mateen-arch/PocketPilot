"""PocketPilot Input Tracing Tool (Viva Examination Aid).

Inspects and traces machine learning pipeline execution step-by-step for a given user input.
Demonstrates intermediate transformations, feature names, encoding changes, prediction output,
and probabilistic risk assessment for both Battery Drain and Screen Time pipelines.
"""

import argparse
import io
from pathlib import Path
import sys
from typing import Any
import joblib
import pandas as pd

# Bootstrap sys.path with project root and src directory
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for p in (ROOT_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pocketpilot import config
from pocketpilot.inference.formatting import format_probability
from pocketpilot.inference.trace import trace_pipeline
from pocketpilot.risk.probabilistic_risk import assess_budget

STEP_DESCRIPTIONS: dict[str, str] = {
    "FunctionTransformer": "Adds the derived column app_share_of_screen from raw inputs.",
    "ColumnTransformer": "One-hot encodes device_model and passes or scales numeric columns.",
    "XGBRegressor": "Predicts daily resource demand using gradient-boosted decision trees.",
    "RandomForestRegressor": "Predicts daily resource demand using a random forest ensemble.",
    "LinearRegression": "Predicts daily resource demand using linear regression.",
}


def trace_and_print_task(
    task_name: str,
    pipeline: Any,
    risk_bundle: dict[str, Any],
    input_df: pd.DataFrame,
    budget: float,
    unit: str,
    output_stream: io.StringIO | None = None,
) -> str:
    """Run trace_pipeline on a task pipeline and format readable output."""
    lines = []
    lines.append("=" * 75)
    lines.append(f"PIPELINE EXECUTION TRACE: {task_name.upper()} ({unit})")
    lines.append("=" * 75)
    lines.append("Raw Input DataFrame:")
    for col, val in input_df.iloc[0].items():
        lines.append(f"  - {col}: {val}")
    lines.append("")

    traces = trace_pipeline(pipeline, input_df)

    for i, tr in enumerate(traces, start=1):
        lines.append(f"Stage {i}: Step '{tr['step_name']}' [{tr['class_name']}]")
        # Explanation
        expl = STEP_DESCRIPTIONS.get(
            tr["class_name"],
            f"Transforms input data via {tr['class_name']}."
        )
        if tr["step_name"] == "model":
            expl = f"Predicts daily {task_name} in {unit} from processed features."
        lines.append(f"  Action:       {expl}")
        lines.append(f"  Output Type:  {tr['output_type']}")
        lines.append(f"  Output Shape: {tr['output_shape']}")
        
        feat_names = tr.get("feature_names")
        if feat_names:
            lines.append(f"  Features ({len(feat_names)}): {feat_names}")

        row_vals = tr.get("output_row", [{}])[0]
        lines.append("  Output Values (Row 0):")
        for k, v in row_vals.items():
            lines.append(f"    * {k}: {v}")
        lines.append("-" * 75)

    # Final prediction and risk assessment
    pred_val = float(traces[-1]["output_row"][0]["prediction"])
    lines.append(f"FINAL PREDICTION ({task_name}): {pred_val:.2f} {unit}")

    if risk_bundle and budget is not None:
        risk_res = assess_budget(risk_bundle, pred_val, budget)
        p_over = risk_res["p_overrun"]
        low_80, high_80 = risk_res["interval_80"]
        lines.append("")
        lines.append("BUDGET & UNCERTAINTY RISK ASSESSMENT:")
        lines.append(f"  Operational Budget: {budget:.1f} {unit}")
        lines.append(f"  Categorical Risk:   {risk_res['level']}")
        lines.append(f"  P(Overrun):         {format_probability(p_over)} ({p_over:.4f})")
        lines.append(f"  80% Interval:       {low_80:.1f} to {high_80:.1f} {unit}")

    lines.append("=" * 75)
    lines.append("")

    text_block = "\n".join(lines)
    print(text_block)
    return text_block


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace PocketPilot machine-learning inference pipelines step-by-step."
    )
    parser.add_argument("--device_model", type=str, default="Samsung Galaxy S21", help="Device model name")
    parser.add_argument("--app_usage_min", type=float, default=250.0, help="Expected app usage in minutes")
    parser.add_argument("--screen_on_hours", type=float, default=5.0, help="Expected screen-on hours")
    parser.add_argument("--apps_installed", type=int, default=50, help="Installed apps count")
    parser.add_argument("--budget_battery", type=float, default=1800.0, help="Battery budget in mAh")
    parser.add_argument("--budget_screen", type=float, default=6.0, help="Screen time budget in hours")

    args = parser.parse_args()

    # Load pipelines and risk bundles
    battery_pipe = joblib.load(config.MODEL_PATHS["battery"])
    screen_pipe = joblib.load(config.MODEL_PATHS["screen_time"])
    risk_bundles = joblib.load(config.RISK_MODEL_PATH)

    # 1. Battery inputs (device_model, app_usage_min, screen_on_hours, apps_installed)
    df_bat = pd.DataFrame([{
        "device_model": args.device_model,
        "app_usage_min": args.app_usage_min,
        "screen_on_hours": args.screen_on_hours,
        "apps_installed": args.apps_installed,
    }])

    # 2. Screen time inputs (device_model, app_usage_min, apps_installed) - NO screen_on_hours
    df_screen = pd.DataFrame([{
        "device_model": args.device_model,
        "app_usage_min": args.app_usage_min,
        "apps_installed": args.apps_installed,
    }])

    full_output = []
    header = (
        "=======================================================================\n"
        "POCKETPILOT INFERENCE PIPELINE TRACE REPORT (VIVA AUDIT TOOL)\n"
        "=======================================================================\n"
        f"Device: {args.device_model} | App Usage: {args.app_usage_min} min | Screen: {args.screen_on_hours} h | Apps: {args.apps_installed}\n\n"
    )
    print(header)
    full_output.append(header)

    bat_text = trace_and_print_task(
        task_name="battery",
        pipeline=battery_pipe,
        risk_bundle=risk_bundles.get("battery", {}),
        input_df=df_bat,
        budget=args.budget_battery,
        unit="mAh",
    )
    full_output.append(bat_text)

    scr_text = trace_and_print_task(
        task_name="screen_time",
        pipeline=screen_pipe,
        risk_bundle=risk_bundles.get("screen_time", {}),
        input_df=df_screen,
        budget=args.budget_screen,
        unit="hours",
    )
    full_output.append(scr_text)

    # Save default run output to reports/testing_evidence/trace_example.txt
    evidence_dir = ROOT_DIR / "reports" / "testing_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    out_file = evidence_dir / "trace_example.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(full_output))
    print(f"[Trace Output Saved] -> {out_file.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
