#!/usr/bin/env python3
"""Compare DS/Qwen selected Label sets with historical IDs on the same units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.label_set_comparison import compare_label_sets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--legacy-snapshot", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=100000)
    args = parser.parse_args()
    report = compare_label_sets(
        units_path=args.units,
        predictions_path=args.predictions,
        labels_path=args.labels,
        legacy_snapshot_path=args.legacy_snapshot,
        output_dir=args.run_dir,
        expected_units=args.expected_units,
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
