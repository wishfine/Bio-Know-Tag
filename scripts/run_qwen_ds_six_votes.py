#!/usr/bin/env python3
"""Run three streaming votes on Qwen and three on DS simultaneously per shard."""

import argparse
import json
from pathlib import Path

from bio_know_tag.six_vote_runner import run_six_vote_shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--qwen-endpoint", default="http://172.22.0.35:9204/v1/chat/completions")
    parser.add_argument("--qwen-model", default="qwen3.8-27b-fp8")
    parser.add_argument("--ds-endpoint", default="http://172.22.0.35:9205/v1/chat/completions")
    parser.add_argument("--ds-model", default="ds-v4-flash")
    parser.add_argument("--workers-per-vote", type=int, default=25)
    parser.add_argument("--qwen-max-tokens", type=int, default=1024)
    parser.add_argument("--ds-max-tokens", type=int, default=512)
    parser.add_argument("--qwen-timeout", type=float, default=600)
    parser.add_argument("--ds-timeout", type=float, default=300)
    parser.add_argument("--qwen-retries", type=int, default=3)
    parser.add_argument("--ds-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--skip-model-preflight", action="store_true")
    args = parser.parse_args()
    result = run_six_vote_shards(
        args.shard_root, args.labels,
        qwen_endpoint=args.qwen_endpoint, qwen_model=args.qwen_model,
        ds_endpoint=args.ds_endpoint, ds_model=args.ds_model,
        workers_per_vote=args.workers_per_vote,
        qwen_max_tokens=args.qwen_max_tokens, ds_max_tokens=args.ds_max_tokens,
        qwen_timeout=args.qwen_timeout, ds_timeout=args.ds_timeout,
        qwen_retries=args.qwen_retries, ds_retries=args.ds_retries,
        retry_delay=args.retry_delay,
        preflight=not args.skip_model_preflight,
        max_shards=args.max_shards,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
