#!/usr/bin/env python3
"""Build a risk-stratified review package from paired 100k adjudication runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.adjudication_review_sample import (
    build_adjudication_review_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--top25-predictions", type=Path, required=True)
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--positive-per-label", type=Path, required=True)
    parser.add_argument("--boundary-assessments", type=Path, required=True)
    parser.add_argument("--strategies", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--high-count", type=int, default=20)
    parser.add_argument("--medium-count", type=int, default=10)
    parser.add_argument("--stable-count", type=int, default=5)
    parser.add_argument("--seed", default="adjudication-review-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-adjudication-review-sample"
    )
    report = build_adjudication_review_sample(
        args.units,
        args.top25_predictions,
        args.legacy_predictions,
        args.labels,
        args.positive_per_label,
        args.boundary_assessments,
        run_dir,
        strategies_path=args.strategies,
        high_count=args.high_count,
        medium_count=args.medium_count,
        stable_count=args.stable_count,
        seed=args.seed,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
