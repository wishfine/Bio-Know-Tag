#!/usr/bin/env python3
"""Split paired Top25 and Top25+legacy adjudication into A/B/C groups."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.legacy_perturbation import analyze_legacy_candidate_perturbation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top25-candidates", required=True)
    parser.add_argument("--legacy-candidates", required=True)
    parser.add_argument("--top25-predictions", required=True)
    parser.add_argument("--legacy-predictions", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--luna-reviews")
    parser.add_argument("--run-dir")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-legacy-perturbation"
    )
    report = analyze_legacy_candidate_perturbation(
        top25_candidates_path=args.top25_candidates,
        legacy_candidates_path=args.legacy_candidates,
        top25_predictions_path=args.top25_predictions,
        legacy_predictions_path=args.legacy_predictions,
        labels_path=args.labels,
        luna_reviews_path=args.luna_reviews,
        run_dir=run_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

