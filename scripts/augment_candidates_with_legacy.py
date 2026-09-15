#!/usr/bin/env python3
"""Append valid current-458 legacy IDs as a candidate source after recall validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.legacy_validation import augment_candidates_with_legacy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-legacy-additions", type=int)
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime("%Y%m%d-%H%M%S-legacy-augmented")
    report = augment_candidates_with_legacy(
        args.units,
        args.candidates,
        args.labels,
        run_dir,
        max_legacy_additions=args.max_legacy_additions,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
