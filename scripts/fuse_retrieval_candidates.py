#!/usr/bin/env python3
"""Fuse BM25 and Dense candidates with configurable reserved quotas."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.retrieval import run_hybrid_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-candidates", type=Path, required=True)
    parser.add_argument("--dense-candidates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--sparse-quota", type=int, default=18)
    parser.add_argument("--dense-quota", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if args.sparse_quota < 0 or args.dense_quota < 0:
        raise SystemExit("quotas must be non-negative")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-hybrid-recall"
    )
    report = run_hybrid_retrieval(
        args.sparse_candidates,
        args.dense_candidates,
        run_dir,
        top_k=args.top_k,
        sparse_quota=args.sparse_quota,
        dense_quota=args.dense_quota,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
