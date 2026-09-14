#!/usr/bin/env python3
"""Use DS and all 458 Label names/@ paths to produce coarse candidates."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from bio_know_tag.ds import DSClient
from bio_know_tag.retrieval import run_ds_coarse_recall


DEFAULT_ENDPOINTS = (
    "http://172.22.0.35:9092/v1/chat/completions",
    "http://172.22.0.35:9093/v1/chat/completions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    return parser.parse_args()


def configured_endpoints(cli_endpoints: list[str] | None) -> list[str]:
    if cli_endpoints:
        return cli_endpoints
    environment = [os.getenv("DS1"), os.getenv("DS2")]
    return [value for value in environment if value] or list(DEFAULT_ENDPOINTS)


def main() -> int:
    args = parse_args()
    for option, value in (
        ("--top-k", args.top_k),
        ("--batch-size", args.batch_size),
        ("--retries", args.retries),
        ("--max-tokens", args.max_tokens),
    ):
        if value < 1:
            raise SystemExit(f"{option} must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-ds-coarse-recall"
    )
    client = DSClient(
        configured_endpoints(args.endpoints),
        args.model,
        timeout=args.timeout,
        retries=args.retries,
    )
    report = run_ds_coarse_recall(
        args.units,
        args.labels,
        run_dir,
        client,
        model=args.model,
        top_k=args.top_k,
        batch_size=args.batch_size,
        limit=args.limit,
        max_tokens=args.max_tokens,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["success"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
