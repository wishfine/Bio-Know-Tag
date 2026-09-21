#!/usr/bin/env python3
"""Generate detailed more/fewer/replacement analysis for one DS stability run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.ds_stability_analysis import analyze_stability_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--baseline", help="condition name; defaults to first manifest condition")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = analyze_stability_run(
        args.run_dir,
        labels_path=args.labels,
        baseline=args.baseline,
        output_dir=args.output_dir,
        require_complete=args.require_complete,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir or args.run_dir / "detailed-analysis"),
        "baseline": report["baseline"],
        "completion": report["completion"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
