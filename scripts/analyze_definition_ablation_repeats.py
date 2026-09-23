#!/usr/bin/env python3
"""Separate Qwen definition effects from same-prompt run-to-run variation."""

import argparse
import json
from pathlib import Path

from bio_know_tag.definition_ablation_repeat import analyze_repeated_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--name-1", type=Path, required=True)
    parser.add_argument("--name-2", type=Path, required=True)
    parser.add_argument("--definition-1", type=Path, required=True)
    parser.add_argument("--definition-2", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_repeated_ablation(
        sample_path=args.sample,
        result_paths={
            "name_1": args.name_1,
            "name_2": args.name_2,
            "definition_1": args.definition_1,
            "definition_2": args.definition_2,
        },
        output_dir=args.run_dir,
    )
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0 if report["complete_pairs"] == report["sample_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
