#!/usr/bin/env python3
"""Stage 1: ask DS to explain every label using only its name."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from bio_know_tag.ds import (
    DSClient,
    append_evidence,
    build_stage1_prompt,
    load_completed_ids,
    parse_json_content,
    read_jsonl,
    summarize_evidence,
    validate_stage1_result,
    write_json_atomic,
)


DEFAULT_ENDPOINTS = (
    "http://172.22.0.35:9092/v1/chat/completions",
    "http://172.22.0.35:9093/v1/chat/completions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def configured_endpoints(cli_endpoints: list[str] | None) -> list[str]:
    if cli_endpoints:
        return cli_endpoints
    environment = [os.getenv("DS1"), os.getenv("DS2")]
    return [value for value in environment if value] or list(DEFAULT_ENDPOINTS)


def main() -> int:
    args = parse_args()
    labels = read_jsonl(args.labels)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be a positive integer")
        labels = labels[: args.limit]
    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-stage1"
    )
    evidence_path = run_dir / "stage1.evidence.jsonl"
    report_path = run_dir / "stage1.report.json"
    completed = load_completed_ids(evidence_path)
    client = DSClient(
        configured_endpoints(args.endpoints),
        args.model,
        timeout=args.timeout,
        retries=args.retries,
    )

    for index, label in enumerate(labels, start=1):
        label_id = str(label["label_id"])
        if label_id in completed:
            continue
        prompt = build_stage1_prompt(label["label_name"])
        record = {
            "stage": "label_self_explain",
            "label_id": label_id,
            "label_name": label["label_name"],
            "prompt": prompt,
            "model": args.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
            "error": None,
        }
        try:
            response = client.chat(
                [
                    {"role": "system", "content": "你是严谨的高中生物知识点判别器。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=args.max_tokens,
            )
            record.update(
                {
                    "raw_response": response.content,
                    "parsed_response": validate_stage1_result(
                        parse_json_content(response.content)
                    ),
                    "endpoint": response.endpoint,
                    "attempts": response.attempts,
                    "latency_seconds": response.latency_seconds,
                }
            )
        except Exception as exc:  # Persist per-label failure and continue the batch.
            record["error"] = f"{type(exc).__name__}: {exc}"
        append_evidence(evidence_path, record)
        report = summarize_evidence(
            evidence_path, (str(item["label_id"]) for item in labels)
        )
        write_json_atomic(report_path, report)
        print(
            f"[{index}/{len(labels)}] {label['label_name']} "
            f"{'ERROR' if record['error'] else 'OK'}",
            flush=True,
        )

    report = summarize_evidence(
        evidence_path, (str(item["label_id"]) for item in labels)
    )
    write_json_atomic(report_path, report)
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["processed"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

