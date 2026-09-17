#!/usr/bin/env python3
"""Export per-Label JSON packages for biology teacher review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.teacher_review_export import export_teacher_review_packages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--positive-results", type=Path, required=True)
    parser.add_argument("--positive-per-label", type=Path, required=True)
    parser.add_argument("--corrected-assessments", type=Path, required=True)
    parser.add_argument("--hard-negative-samples", type=Path, required=True)
    parser.add_argument("--hard-negative-results", type=Path, required=True)
    parser.add_argument("--colabel-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--examples-per-side", type=int, default=10)
    parser.add_argument("--high-boundary-threshold", type=float, default=0.15)
    parser.add_argument("--min-positive-count", type=int, default=300)
    parser.add_argument("--min-valid-negative-count", type=int, default=20)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-teacher-review-packages"
    )
    report = export_teacher_review_packages(
        labels_path=args.labels,
        strategies_path=args.strategies,
        positive_results_path=args.positive_results,
        positive_per_label_path=args.positive_per_label,
        corrected_assessments_path=args.corrected_assessments,
        hard_negative_samples_path=args.hard_negative_samples,
        hard_negative_results_path=args.hard_negative_results,
        colabel_results_path=args.colabel_results,
        output_dir=output_dir,
        examples_per_side=args.examples_per_side,
        high_boundary_threshold=args.high_boundary_threshold,
        min_positive_count=args.min_positive_count,
        min_valid_negative_count=args.min_valid_negative_count,
    )
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
