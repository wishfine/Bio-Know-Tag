#!/usr/bin/env python3
"""Clean raw question JSONL and aggregate sub-questions into parent records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.questions import process_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-preprocess"
    )
    report = process_jsonl(
        args.input,
        run_dir / "questions.jsonl",
        run_dir / "report.json",
        limit=args.limit,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

