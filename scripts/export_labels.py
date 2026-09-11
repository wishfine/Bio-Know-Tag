#!/usr/bin/env python3
"""Export the teacher-authored taxonomy workbook as validated JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.labels import load_label_workbook, write_label_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .xlsx file")
    parser.add_argument(
        "--output", type=Path, default=Path("configs/labels.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("configs/labels.report.json")
    )
    parser.add_argument("--expect-count", type=int, default=458)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_label_workbook(args.input)
    if len(records) != args.expect_count:
        raise SystemExit(
            f"label count mismatch: expected {args.expect_count}, got {len(records)}"
        )
    report = write_label_outputs(records, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

