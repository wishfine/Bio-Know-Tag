#!/usr/bin/env python3
"""Build balanced sibling-Label hard negatives from completed positive results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.hard_negatives import build_verified_sibling_hard_negatives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-tasks", type=Path, required=True)
    parser.add_argument("--positive-results", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--negatives-per-label", type=int, default=50)
    parser.add_argument("--min-source-score", type=float, default=0.80)
    parser.add_argument("--seed", default="verified-sibling-hard-negative-v1")
    parser.add_argument("--allow-partial-positive-results", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-verified-hard-negatives"
    )
    report = build_verified_sibling_hard_negatives(
        args.positive_tasks,
        args.positive_results,
        args.units,
        args.labels,
        run_dir,
        negatives_per_label=args.negatives_per_label,
        min_source_score=args.min_source_score,
        seed=args.seed,
        require_complete=not args.allow_partial_positive_results,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

