#!/usr/bin/env python3
"""Run resumable DS scored judgments for question-Label definition coverage."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from bio_know_tag.boundary_judge import run_boundary_judge
from bio_know_tag.ds import DSClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2)
    parser.add_argument("--request-interval", type=float, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    endpoints = args.endpoints or [value for value in (os.getenv("DS1"), os.getenv("DS2")) if value]
    if not endpoints:
        raise SystemExit("provide --endpoint or set DS1/DS2")
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime("%Y%m%d-%H%M%S-label-boundary-judge")
    client = DSClient(
        endpoints,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
    )
    report = run_boundary_judge(
        args.samples,
        args.labels,
        run_dir,
        client,
        model=args.model,
        max_tokens=args.max_tokens,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["success"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
