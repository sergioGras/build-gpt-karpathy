"""The CLI and notebooks share the same configuration and experiment functions."""

import argparse
import json
import sys

from .config import load_experiment
from .runs import prepare_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and prepare guided GPT-2 experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare-run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to an experiment JSON file")
        if command == "prepare-run":
            subparser.add_argument("--output-dir", default="runs", help="Run directory (relative to launch folder)")
    arguments = parser.parse_args(argv)
    try:
        experiment = load_experiment(arguments.config)
        if arguments.command == "plan":
            print(json.dumps(experiment.plan(), indent=2))
        else:
            run_dir = prepare_run(experiment, arguments.output_dir)
            print(f"Prepared: {run_dir}\nNo training has run.")
    except (ValueError, OSError) as error:
        print(f"gpt2-lab: {error}", file=sys.stderr)
        return 1
    return 0
