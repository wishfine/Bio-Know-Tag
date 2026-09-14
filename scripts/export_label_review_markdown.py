#!/usr/bin/env python3
"""Export the per-label DS/GPT review ledger to Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.label_review_markdown import read_jsonl, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("configs/label_strategies.review2.jsonl"),
        help="Second-review JSONL ledger.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/label-review-table.md"),
        help="Destination Markdown path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_markdown(read_jsonl(args.input), args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
