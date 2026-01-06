"""
Cyber Threat Insight
Main Project Entry Point

This script orchestrates the full cybersecurity analytics lifecycle:
- Model development
- Inference
- Production simulation
- Cyber attack simulation
- Executive dashboard visualization
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


PIPELINE_STAGES = {
    "dev": "model_dev/stacked_model/stacked_anomaly_detection_classifier.py",
    "inference": "model_inference/model_inference.py",
    "production": "production/stacked_ad_classifier_prod.py",
    "attack": "cyber_attack_insight/attack_simulation_v02.py",
    "dashboard": "cyber_attack_insight/attacks_executive_dashboard_v02.py",
}


def run_stage(stage: str):
    script_path = PROJECT_ROOT / PIPELINE_STAGES[stage]

    if not script_path.exists():
        raise FileNotFoundError(f"❌ Script not found: {script_path}")

    print(f"\n🚀 Running stage: {stage.upper()}")
    print(f"📄 Script: {script_path}\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True
    )

    if result.returncode == 0:
        print(f"✅ Stage '{stage}' completed successfully\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cyber Threat Insight – End-to-End Cybersecurity Analytics Pipeline"
    )

    parser.add_argument(
        "--stage",
        choices=list(PIPELINE_STAGES.keys()) + ["all"],
        default="all",
        help="Pipeline stage to run"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n==============================")
    print("🛡️  Cyber Threat Insight")
    print("==============================")

    if args.stage == "all":
        for stage in PIPELINE_STAGES:
            run_stage(stage)
    else:
        run_stage(args.stage)

    print("🎉 Pipeline execution completed.")


if __name__ == "__main__":
    main()
