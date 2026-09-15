#!/usr/bin/env python3
"""Evaluate Recall@K against valid current-taxonomy historical IDs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.legacy_validation import evaluate_legacy_recall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--k", type=int, action="append", dest="ks")
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime("%Y%m%d-%H%M%S-legacy-recall-eval")
    report = evaluate_legacy_recall(
        args.units, args.candidates, args.labels, run_dir, ks=tuple(args.ks or (5, 10, 20, 25))
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["missing_candidate_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
