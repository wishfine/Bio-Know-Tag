#!/usr/bin/env python3
"""Stream one complete aligned question/candidate file through bounded adjudication."""

import argparse
import json
from pathlib import Path

from bio_know_tag.ds import DSClient
from bio_know_tag.full_adjudication import run_full_adjudication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1)
    args = parser.parse_args()
    client = DSClient(
        [args.endpoint], args.model, timeout=args.timeout,
        retries=args.retries, retry_delay=args.retry_delay,
        request_interval=0,
    )
    report = run_full_adjudication(
        args.units, args.candidates, args.labels, args.run_dir,
        client, model=args.model, workers=args.workers,
        max_tokens=args.max_tokens,
        request_config={
            "endpoint": args.endpoint,
            "timeout": args.timeout,
            "retries": args.retries,
            "retry_delay": args.retry_delay,
            "request_interval": 0,
            "enable_thinking": None,
        },
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0 if report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
