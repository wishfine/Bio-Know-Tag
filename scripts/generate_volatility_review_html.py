#!/usr/bin/env python3
"""Generate a standalone HTML for teacher review of label volatility."""

from __future__ import annotations

import argparse
import json

from bio_know_tag.volatility_review_html import build_volatility_review_html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--labels", default="configs/labels.jsonl")
    parser.add_argument("--image-context")
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    report = build_volatility_review_html(
        group_paths=args.groups,
        units_path=args.units,
        labels_path=args.labels,
        image_context_path=args.image_context,
        output_html_path=args.output_html,
        output_jsonl_path=args.output_jsonl,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

