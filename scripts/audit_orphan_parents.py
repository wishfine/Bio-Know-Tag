#!/usr/bin/env python3
"""Audit parent IDs referenced by children but absent as raw question rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.audit import audit_orphan_parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-orphan-audit"
    )
    report = audit_orphan_parents(
        args.raw,
        args.processed,
        run_dir / "orphan_parents.jsonl",
        run_dir / "report.json",
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    blocking_errors = (
        report["raw_error"]
        + report["processed_error"]
        + report["orphan_records_missing"]
    )
    return 0 if blocking_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

