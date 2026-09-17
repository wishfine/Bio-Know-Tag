#!/usr/bin/env python3
"""Combine positive coverage with co-label-corrected hard-negative results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.co_label_judge import combine_corrected_boundary_assessments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-per-label", type=Path, required=True)
    parser.add_argument("--negative-per-label", type=Path, required=True)
    parser.add_argument("--colabel-per-target", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-corrected-boundary-combined"
    )
    report = combine_corrected_boundary_assessments(
        args.positive_per_label,
        args.negative_per_label,
        args.colabel_per_target,
        run_dir,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
