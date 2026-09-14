#!/usr/bin/env python3
"""Run local character n-gram BM25 candidate retrieval on Pilot units."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.retrieval import run_sparse_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-sparse-recall"
    )
    report = run_sparse_retrieval(
        args.units,
        args.labels,
        run_dir,
        top_k=args.top_k,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
