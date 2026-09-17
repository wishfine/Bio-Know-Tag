#!/usr/bin/env python3
"""Independently verify Stage-1 selected Labels against definitions."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from bio_know_tag.definition_verifier import run_definition_verifier
from bio_know_tag.ds import DSClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--stage1-predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = args.endpoints or [
        value for value in (os.getenv("DS1"), os.getenv("DS2")) if value
    ]
    if not endpoints:
        raise SystemExit("provide --endpoint or set DS1/DS2")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-definition-verifier"
    )
    client = DSClient(
        endpoints,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
    )
    report = run_definition_verifier(
        args.units,
        args.stage1_predictions,
        args.labels,
        run_dir,
        client,
        model=args.model,
        limit=args.limit,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["error"] == 0 and report["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
