#!/usr/bin/env python3
"""Run one long-form diagnostic adjudication without changing the main pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.adjudication import build_adjudication_prompt
from bio_know_tag.ds import DSClient, parse_json_content


def _find_jsonl(path: Path, key: str, value: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get(key) or "") == value:
                return row
    raise ValueError(f"{value} not found in {path}")


def _read_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[str(row["label_id"])] = row
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--focus-label-id", action="append", default=[])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = args.endpoints or [
        value for value in (os.getenv("DS1"), os.getenv("DS2")) if value
    ]
    if not endpoints:
        raise SystemExit("provide --endpoint or set DS1/DS2")

    unit = _find_jsonl(args.units, "question_id", args.question_id)
    candidate_row = _find_jsonl(
        args.candidates, "question_id", args.question_id
    )
    labels = _read_labels(args.labels)
    candidate_ids = {
        str(candidate["label_id"])
        for candidate in candidate_row.get("candidates") or []
    }
    missing_focus = [
        label_id for label_id in args.focus_label_id if label_id not in candidate_ids
    ]
    if missing_focus:
        raise SystemExit(
            f"focus label is not in this question's candidates: {missing_focus[0]}"
        )

    prompt, code_map = build_adjudication_prompt(
        unit,
        candidate_row.get("candidates") or [],
        labels,
        diagnostic_focus_label_ids=set(args.focus_label_id),
    )
    client = DSClient(
        endpoints,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    response = client.chat(
        [
            {
                "role": "system",
                "content": "你是严谨的高中生物知识点审计器，只输出JSON。",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.max_tokens,
    )
    parsed = parse_json_content(response.content)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "raw_response.txt").write_text(response.content, encoding="utf-8")
    result = {
        "question_id": args.question_id,
        "model": args.model,
        "endpoint": response.endpoint,
        "attempts": response.attempts,
        "latency_seconds": response.latency_seconds,
        "usage": response.usage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "candidate_count": len(candidate_ids),
        "candidate_code_map": code_map,
        "focus_label_ids": args.focus_label_id,
        "parsed_response": parsed,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
