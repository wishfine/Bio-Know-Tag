#!/usr/bin/env python3
"""Reconnect audited source parent context and exclude truly orphaned children."""

import argparse
import json
from pathlib import Path

from bio_know_tag.repair_dedup_parent_context import repair_dedup_parent_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--parent-source-audit", type=Path, required=True)
    parser.add_argument("--dedup-raw", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    report = repair_dedup_parent_context(
        args.processed, args.parent_source_audit, args.dedup_raw, args.run_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
