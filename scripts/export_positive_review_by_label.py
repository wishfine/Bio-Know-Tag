#!/usr/bin/env python3
"""Export one readable JSON file per Label from positive-coverage results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.positive_review_export import export_positive_review_by_label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("output") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-positive-review-by-label"
    )
    report = export_positive_review_by_label(
        args.tasks,
        args.results,
        args.labels,
        output_dir,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
