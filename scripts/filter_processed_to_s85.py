#!/usr/bin/env python3
"""Reuse merged preprocessing while retaining exactly the s85 question IDs."""

import argparse
import json
from pathlib import Path

from bio_know_tag.filter_processed_s85 import filter_processed_s85


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--dedup-raw", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--subject", default="生物")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    report = filter_processed_s85(
        args.processed, args.dedup_raw, args.updates, args.run_dir,
        subject=args.subject, progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
