#!/usr/bin/env python3
"""Build a deterministic no-legacy Pilot sample from labeling derivatives."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.pilot import build_pilot_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-units", type=Path, required=True)
    parser.add_argument("--parent-aggregation", type=Path, required=True)
    parser.add_argument("--duplicate-groups", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--target-size", type=int, default=2_500)
    parser.add_argument("--parent-groups", type=int, default=200)
    parser.add_argument("--duplicate-samples", type=int, default=100)
    parser.add_argument("--audit-sample-size", type=int, default=100)
    parser.add_argument("--stratum-sample-size", type=int, default=3)
    parser.add_argument("--seed", default="bio-pilot-no-legacy-v1")
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positive = {
        "--target-size": args.target_size,
        "--parent-groups": args.parent_groups,
        "--duplicate-samples": args.duplicate_samples,
        "--audit-sample-size": args.audit_sample_size,
        "--stratum-sample-size": args.stratum_sample_size,
    }
    for option, value in positive.items():
        if value < 1:
            raise SystemExit(f"{option} must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-pilot-sample"
    )
    report = build_pilot_package(
        args.label_units,
        args.parent_aggregation,
        args.duplicate_groups,
        run_dir,
        target_size=args.target_size,
        parent_groups=args.parent_groups,
        duplicate_groups=args.duplicate_samples,
        audit_sample_size=args.audit_sample_size,
        stratum_sample_size=args.stratum_sample_size,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
