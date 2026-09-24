#!/usr/bin/env python3
"""Run three concurrent streaming Qwen votes per fine-ranking shard."""

import argparse
import json
from pathlib import Path

from bio_know_tag.three_vote_runner import run_three_vote_shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--endpoint", default="http://172.22.0.35:9204/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-27b-fp8")
    parser.add_argument("--workers-per-vote", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--skip-model-preflight", action="store_true")
    args = parser.parse_args()
    report = run_three_vote_shards(
        args.shard_root, args.labels, args.endpoint, args.model,
        workers_per_vote=args.workers_per_vote,
        max_tokens=args.max_tokens, timeout=args.timeout,
        retries=args.retries, retry_delay=args.retry_delay,
        disable_thinking=args.disable_thinking,
        preflight=not args.skip_model_preflight,
        max_shards=args.max_shards,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
