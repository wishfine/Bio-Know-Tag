#!/usr/bin/env python3
"""Materialize common successful predictions from two live evidence files."""

import argparse
from pathlib import Path

from bio_know_tag.adjudication_snapshot import build_paired_adjudication_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--top25-evidence", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--audited-exclusions", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    build_paired_adjudication_snapshot(
        args.units,
        args.top25_evidence,
        args.legacy_evidence,
        args.top25_candidates,
        args.legacy_candidates,
        args.labels,
        args.run_dir,
        audited_exclusions_path=args.audited_exclusions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
