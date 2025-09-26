#!/usr/bin/env python3
"""
Simple wandb wrapper for Time-Series-Library experiments
"""
import wandb
import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run TS experiments with wandb")
    parser.add_argument("--script", type=str, required=True,
                       help="Script to run (e.g., scripts/long_term_forecast/Exchange_script/SO2SPDPolar.sh)")
    parser.add_argument("--name", type=str, default="", help="Experiment name")
    parser.add_argument("--tags", type=str, nargs="*", default=[], help="Tags")
    parser.add_argument("--project", type=str, default="SO2-SPD-Polar-TSF", help="WandB project")
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    # Auto-generate name if not provided
    if not args.name:
        args.name = script_path.stem

    print(f"🚀 Running: {script_path.name}")
    print(f"📊 Project: {args.project}")

    # Initialize wandb
    run = wandb.init(
        project=args.project,
        name=args.name,
        tags=args.tags + [script_path.parent.name]
    )

    try:
        # Run the script
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd="Time-Series-Library",
            capture_output=True,
            text=True,
            check=False
        )

        # Log basic results
        wandb.log({
            "success": result.returncode == 0,
            "exit_code": result.returncode
        })

        # Save logs
        if result.stdout:
            with open("stdout.log", "w") as f:
                f.write(result.stdout)
            wandb.save("stdout.log")

        if result.stderr:
            with open("stderr.log", "w") as f:
                f.write(result.stderr)
            wandb.save("stderr.log")

        if result.returncode == 0:
            print("✅ Success!")
        else:
            print(f"❌ Failed (exit code: {result.returncode})")

        print(f"📈 View at: {run.url}")

    finally:
        run.finish()

if __name__ == "__main__":
    main()