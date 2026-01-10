"""
main.py (Config + Logging + Retry)
Cyber Threat Insight
Main Project Entry Point

This script orchestrates the full cybersecurity analytics lifecycle:
- Model development
- Inference
- Production simulation
- Cyber attack simulation
- Executive dashboard visualization
"""

import yaml
import subprocess
import sys
from pathlib import Path
from utils.logger import setup_logger, timeit
from utils.retry import retry

logger = setup_logger("pipeline")
PROJECT_ROOT = Path(__file__).parent

with open("config/pipeline.yaml") as f:
    CONFIG = yaml.safe_load(f)

PIPELINE = CONFIG["pipeline"]
RETRY_CFG = CONFIG["retry"]


@retry(RETRY_CFG["max_attempts"], RETRY_CFG["delay_seconds"])
@timeit(logger)
def run_stage(stage):
    script = PROJECT_ROOT / PIPELINE[stage]
    subprocess.run([sys.executable, str(script)], check=True)


def main(stage="all"):
    logger.info("Starting Cyber Threat Insight Pipeline")

    if stage == "all":
        for s in PIPELINE:
            run_stage(s)
    else:
        run_stage(stage)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()
    main(args.stage)
