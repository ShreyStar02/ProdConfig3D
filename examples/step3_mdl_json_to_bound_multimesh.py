#!/usr/bin/env python3
"""Example wrapper for step3: curate-multimesh."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run step3 / curate-multimesh")
    parser.add_argument("--source", required=True)
    parser.add_argument("--materials-json", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        str(repo_root / "main.py"),
        "step3",
        "--source", args.source,
        "--materials-json", args.materials_json,
        "--dest", args.dest,
    ]
    if args.name:
        command.extend(["--name", args.name])

    return subprocess.run(command, cwd=str(repo_root), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
