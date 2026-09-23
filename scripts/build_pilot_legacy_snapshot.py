#!/usr/bin/env python3
"""Recover old knw_ids for the blind 100k pilot from full label units."""

import argparse
import json
from pathlib import Path

from bio_know_tag.label_set_comparison import build_legacy_id_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-units", type=Path, required=True)
    parser.add_argument("--full-units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=100000)
    args = parser.parse_args()
    print(json.dumps(build_legacy_id_snapshot(
        sample_units_path=args.sample_units,
        full_units_path=args.full_units,
        output_path=args.output,
        expected_units=args.expected_units,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
