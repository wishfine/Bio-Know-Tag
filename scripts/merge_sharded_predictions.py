#!/usr/bin/env python3
"""Merge complete per-shard fine-ranking predictions for one independent vote."""

import argparse
import json
from pathlib import Path

from bio_know_tag.adjudication_shards import merge_shard_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--vote-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        merge_shard_predictions(args.shard_root, args.vote_name, args.output),
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
