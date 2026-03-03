#!/usr/bin/env python3
"""Example wrapper for combined step23: mdl-json-to-bound-multimesh."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run step23 / mdl-json-to-bound-multimesh")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--nim-rerank", action="store_true")
    parser.add_argument("--no-nim-rerank", action="store_true")
    parser.add_argument("--nim-profile", default=None)
    parser.add_argument("--nim-base-url", default=None)
    parser.add_argument("--nim-auth-mode", default=None)
    args = parser.parse_args()

    if args.nim_rerank and args.no_nim_rerank:
        raise ValueError("Use only one of --nim-rerank or --no-nim-rerank")

    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        str(repo_root / "main.py"),
        "step23",
        "--source", args.source,
        "--dest", args.dest,
    ]
    if args.name:
        command.extend(["--name", args.name])
    if args.top_k is not None:
        command.extend(["--top-k", str(args.top_k)])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.nim_rerank:
        command.append("--nim-rerank")
    if args.no_nim_rerank:
        command.append("--no-nim-rerank")
    if args.nim_profile:
        command.extend(["--nim-profile", args.nim_profile])
    if args.nim_base_url:
        command.extend(["--nim-base-url", args.nim_base_url])
    if args.nim_auth_mode:
        command.extend(["--nim-auth-mode", args.nim_auth_mode])

    return subprocess.run(command, cwd=str(repo_root), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
