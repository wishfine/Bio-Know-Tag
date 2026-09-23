#!/usr/bin/env python3
"""Check whether missing parent IDs exist in the pre-dedup raw question source."""

import argparse
import json
from pathlib import Path

from bio_know_tag.orphan_parent_source import audit_orphan_parent_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orphan-audit", type=Path, required=True)
    parser.add_argument("--original-raw", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    report = audit_orphan_parent_source(
        args.orphan_audit, args.original_raw, args.run_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0 if report["original_raw_malformed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
