#!/usr/bin/env python3
"""Run an independent second review over the per-label strategy ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.ds import read_jsonl
from bio_know_tag.strategy import build_second_review_records, write_second_review_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("configs/label_strategies.jsonl"),
        help="First-pass merged ledger.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/label_strategies.review2.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Defaults to <output>.report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report or args.output.with_name(f"{args.output.stem}.report.json")
    input_records = read_jsonl(args.input)
    records = build_second_review_records(input_records)
    report = write_second_review_outputs(
        records,
        args.output,
        report_path,
        input_path=args.input,
    )
    print(
        json.dumps(
            {"output": str(args.output), "report": str(report_path), **report},
            ensure_ascii=False,
        )
    )
    return 0 if report["processed"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
