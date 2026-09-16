#!/usr/bin/env python3
"""Overlay one subject's update-data rows onto an immutable raw JSONL baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_know_tag.update_merge import merge_question_updates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--subject", default="生物")
    args = parser.parse_args()
    report = merge_question_updates(
        args.base,
        args.updates,
        args.output,
        args.report,
        subject=args.subject,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["base_errors"] == 0 and report["update_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

