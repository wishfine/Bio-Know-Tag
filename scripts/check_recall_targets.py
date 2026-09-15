#!/usr/bin/env python3
"""Check whether named Labels occur in candidate lists for selected questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="question_id=exact label name; repeat for multiple targets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = {}
    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["question_id"])] = row
    checks = []
    for raw in args.target:
        if "=" not in raw:
            raise SystemExit(f"invalid --target: {raw!r}")
        question_id, label_name = raw.split("=", 1)
        candidates = (rows.get(question_id) or {}).get("candidates") or []
        match = next(
            (item for item in candidates if item.get("label_name") == label_name),
            None,
        )
        checks.append(
            {
                "question_id": question_id,
                "label_name": label_name,
                "hit": match is not None,
                "rank": (match or {}).get("candidate_rank", (match or {}).get("rank")),
                "candidate_count": len(candidates),
            }
        )
    result = {
        "candidates": str(args.candidates),
        "targets": len(checks),
        "hits": sum(item["hit"] for item in checks),
        "all_hit": all(item["hit"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_hit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
