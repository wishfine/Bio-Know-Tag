#!/usr/bin/env python3
"""Launch six concurrent full-file votes without creating shards."""

import argparse
import json
from pathlib import Path

from bio_know_tag.full_six_vote_runner import run_full_six_votes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qwen-endpoint", default="http://172.22.0.35:9204/v1/chat/completions")
    parser.add_argument("--qwen-model", default="qwen3.8-27b-fp8")
    parser.add_argument("--ds-endpoint", default="http://172.22.0.35:9205/v1/chat/completions")
    parser.add_argument("--ds-model", default="ds-v4-flash")
    parser.add_argument("--workers-per-vote", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--qwen-timeout", type=float, default=600)
    parser.add_argument("--ds-timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--skip-model-preflight", action="store_true")
    args = parser.parse_args()
    report = run_full_six_votes(
        args.units, args.candidates, args.labels, args.run_dir,
        qwen_endpoint=args.qwen_endpoint, qwen_model=args.qwen_model,
        ds_endpoint=args.ds_endpoint, ds_model=args.ds_model,
        workers_per_vote=args.workers_per_vote, max_tokens=args.max_tokens,
        qwen_timeout=args.qwen_timeout, ds_timeout=args.ds_timeout,
        retries=args.retries, retry_delay=args.retry_delay,
        preflight=not args.skip_model_preflight,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
