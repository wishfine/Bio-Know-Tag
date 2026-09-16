#!/usr/bin/env python3
"""Join image URLs by question ID and audit text-only labeling eligibility."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bio_know_tag.image_context import audit_label_unit_image_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-image-context-audit"
    )
    report = audit_label_unit_image_context(
        args.units,
        args.image_map,
        run_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

