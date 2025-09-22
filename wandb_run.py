import wandb
import subprocess
import sys
from pathlib import Path
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run time series experiments with wandb tracking")
    parser.add_argument("--script_path", type=str, required=True, 
                       help="Path to the shell script to execute (e.g., Time-Series-Library/scripts/anomaly_detection/MSL/Autoformer.sh)")
    parser.add_argument("--project", type=str, default="my-awesome-project")
    parser.add_argument("--entity", type=str, default="viresh-georgia-institute-of-technology")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--name", type=str, default="", help="Run name for wandb")
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        notes=args.notes,
        name=args.name
    )
    

    run.finish()

if __name__ == "__main__":
    main()