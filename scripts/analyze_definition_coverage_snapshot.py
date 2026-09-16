#!/usr/bin/env python3
"""Analyze a copied partial/full definition-coverage results JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.coverage_analysis import analyze_coverage_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--samples-per-band", type=int, default=3)
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-definition-coverage-analysis"
    )
    report = analyze_coverage_snapshot(
        args.tasks,
        args.results,
        args.labels,
        run_dir,
        samples_per_band=args.samples_per_band,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

