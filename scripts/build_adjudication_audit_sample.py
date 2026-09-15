#!/usr/bin/env python3
"""Select aligned units and candidates for DS adjudication plus human audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.audit_sample import build_adjudication_audit_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--representative-size", type=int, default=200)
    parser.add_argument("--seed", default="adjudication-audit-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-adjudication-audit-sample"
    )
    report = build_adjudication_audit_sample(
        args.units,
        args.candidates,
        run_dir,
        sample_size=args.sample_size,
        representative_size=args.representative_size,
        seed=args.seed,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
