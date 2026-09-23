#!/usr/bin/env python3
"""Aggregate parent labels from one mixed-unit prediction stream."""

import argparse
import json
from pathlib import Path

from bio_know_tag.unified_parent_aggregation import aggregate_unified_parent_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-aggregation", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_unified_parent_predictions(
        args.parent_aggregation, args.units, args.predictions, args.output,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
