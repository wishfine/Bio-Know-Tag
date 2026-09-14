#!/usr/bin/env python3
"""Compare BM25 and Dense Pilot candidates and export disagreement samples."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.retrieval import compare_candidate_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--sparse-candidates", type=Path, required=True)
    parser.add_argument("--dense-candidates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.sample_size < 1:
        raise SystemExit("--top-k and --sample-size must be positive")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-retrieval-compare"
    )
    report = compare_candidate_runs(
        args.units,
        args.sparse_candidates,
        args.dense_candidates,
        run_dir,
        top_k=args.top_k,
        sample_size=args.sample_size,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
