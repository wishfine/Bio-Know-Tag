#!/usr/bin/env python3
"""Sample legacy-positive and optional sibling-negative question-Label pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.boundary_judge import build_boundary_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--positive-per-label", type=int, default=5)
    parser.add_argument("--negative-per-label", type=int, default=0)
    parser.add_argument("--seed", default="label-boundary-v1")
    parser.add_argument("--unit-type", action="append", dest="unit_types")
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime("%Y%m%d-%H%M%S-label-boundary-sample")
    report = build_boundary_sample(
        args.units,
        args.labels,
        run_dir,
        positive_per_label=args.positive_per_label,
        negative_per_label=args.negative_per_label,
        seed=args.seed,
        unit_types=set(args.unit_types) if args.unit_types else None,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
