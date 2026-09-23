#!/usr/bin/env python3
"""Split one unified retrieval stream into aligned bounded-memory fine-ranking shards."""

import argparse
import json
from pathlib import Path

from bio_know_tag.adjudication_shards import shard_adjudication_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args()
    report = shard_adjudication_inputs(
        args.units, args.candidates, args.run_dir, shard_size=args.shard_size
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
