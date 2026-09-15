#!/usr/bin/env python3
"""Build a deterministic large recall-validation sample from legacy weak labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.legacy_validation import build_legacy_recall_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sample-rate", type=float, default=0.05)
    parser.add_argument("--seed", default="legacy-recall-v1")
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime("%Y%m%d-%H%M%S-legacy-recall-sample")
    report = build_legacy_recall_sample(
        args.units, args.labels, run_dir, sample_rate=args.sample_rate, seed=args.seed
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
