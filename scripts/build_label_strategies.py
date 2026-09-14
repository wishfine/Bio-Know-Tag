#!/usr/bin/env python3
"""Merge teacher taxonomy, Stage 1, and Stage 2 into one per-label ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.ds import read_jsonl
from bio_know_tag.strategy import (
    build_strategy_records,
    latest_successful,
    load_reference_workbook,
    write_strategy_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--stage1-evidence", type=Path, required=True)
    parser.add_argument("--stage2-evidence", type=Path, required=True)
    parser.add_argument(
        "--reference-workbook",
        type=Path,
        help="Optional mentor strategy workbook; it is advisory and preserved as evidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/label_strategies.jsonl"),
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
    labels = read_jsonl(args.labels)
    stage1_records = read_jsonl(args.stage1_evidence)
    stage2_records = read_jsonl(args.stage2_evidence)
    stage1_by_id = latest_successful(stage1_records)
    stage2_by_id = latest_successful(stage2_records)
    reference_by_id = {}
    issues_by_name = {}
    if args.reference_workbook:
        reference_by_id, issues_by_name = load_reference_workbook(args.reference_workbook)
    records = build_strategy_records(
        labels,
        stage1_by_id,
        stage2_by_id,
        reference_by_id=reference_by_id,
        issues_by_name=issues_by_name,
    )
    report = write_strategy_outputs(
        records,
        args.output,
        report_path,
        input_paths={
            "labels": args.labels,
            "stage1_evidence": args.stage1_evidence,
            "stage2_evidence": args.stage2_evidence,
            **({"reference_workbook": args.reference_workbook} if args.reference_workbook else {}),
        },
    )
    print(json.dumps({"output": str(args.output), "report": str(report_path), **report}, ensure_ascii=False))
    return 0 if report["processed"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
