#!/usr/bin/env python3
"""Cross-encoder reranking over the union of sparse and dense candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.reranker import TransformerCrossEncoderReranker
from bio_know_tag.retrieval import run_candidate_reranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--sparse-candidates", type=Path, required=True)
    parser.add_argument("--dense-candidates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", default="BAAI/bge-reranker-base")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sparse-pool", type=int, default=30)
    parser.add_argument("--dense-pool", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for option, value in (
        ("--sparse-pool", args.sparse_pool),
        ("--dense-pool", args.dense_pool),
        ("--top-k", args.top_k),
        ("--batch-size", args.batch_size),
        ("--max-length", args.max_length),
    ):
        if value < 1:
            raise SystemExit(f"{option} must be positive")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-reranked-recall"
    )
    reranker = TransformerCrossEncoderReranker(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        use_fp16=not args.fp32,
    )
    report = run_candidate_reranking(
        args.units,
        args.labels,
        args.sparse_candidates,
        args.dense_candidates,
        run_dir,
        reranker,
        sparse_pool=args.sparse_pool,
        dense_pool=args.dense_pool,
        top_k=args.top_k,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
