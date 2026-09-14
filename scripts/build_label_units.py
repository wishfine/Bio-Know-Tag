#!/usr/bin/env python3
"""Build labeling units and dry-run routing artifacts without calling DS."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.label_units import build_labeling_derivatives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Preprocessed questions.jsonl")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("configs/label_strategies.review2.jsonl"),
    )
    parser.add_argument(
        "--orphan-audit",
        type=Path,
        required=True,
        help="orphan_parents.jsonl produced by audit_orphan_parents.py",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process this many top-level preprocessed records (smoke test)",
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-label-units"
    )
    report = build_labeling_derivatives(
        args.input,
        args.labels,
        args.orphan_audit,
        run_dir,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
