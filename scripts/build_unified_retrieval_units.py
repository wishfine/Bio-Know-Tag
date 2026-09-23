#!/usr/bin/env python3
"""Merge child/standalone and parent-material units into one retrieval stream."""

import argparse
import json
from pathlib import Path

from bio_know_tag.full_retrieval_units import build_unified_retrieval_units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--parent-aggregation", type=Path, required=True)
    parser.add_argument("--keep-ids", type=Path, help="one canonical question_id per line")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    keep_ids = None
    if args.keep_ids is not None:
        keep_ids = set(args.keep_ids.read_text(encoding="utf-8").split())
        if not keep_ids:
            raise SystemExit("--keep-ids is empty")
    report = build_unified_retrieval_units(
        args.units, args.parent_aggregation, args.run_dir,
        keep_question_ids=keep_ids,
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
