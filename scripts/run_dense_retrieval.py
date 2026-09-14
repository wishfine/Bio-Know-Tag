#!/usr/bin/env python3
"""Run Transformer dense candidate retrieval on Pilot units."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.dense import TransformerDenseRetriever
from bio_know_tag.retrieval import run_dense_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no-query-instruction", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for option, value in (
        ("--top-k", args.top_k),
        ("--batch-size", args.batch_size),
        ("--max-length", args.max_length),
    ):
        if value < 1:
            raise SystemExit(f"{option} must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-dense-recall"
    )
    encoder = TransformerDenseRetriever(
        args.model,
        device=args.device,
        encode_batch_size=args.batch_size,
        max_length=args.max_length,
        query_instruction="" if args.no_query_instruction else "为这个句子生成表示以用于检索相关文章：",
        local_files_only=args.local_files_only,
        use_fp16=not args.fp32,
    )
    report = run_dense_retrieval(
        args.units,
        args.labels,
        run_dir,
        encoder,
        top_k=args.top_k,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["processed"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
