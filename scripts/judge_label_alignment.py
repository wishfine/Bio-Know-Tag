#!/usr/bin/env python3
"""Stage 2: judge DS name-only explanations against the teacher taxonomy."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from bio_know_tag.ds import (
    DSClient,
    append_evidence,
    build_alignment_prompt,
    classify_alignment,
    load_completed_ids,
    parse_json_content,
    read_jsonl,
    summarize_evidence,
    validate_alignment_result,
    write_json_atomic,
)
from run_label_self_explain import DEFAULT_ENDPOINTS, configured_endpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--stage1-evidence", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--high-score-sample", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=20260911)
    return parser.parse_args()


def successful_stage1(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in read_jsonl(path):
        if not record.get("error") and isinstance(record.get("parsed_response"), dict):
            latest[str(record["label_id"])] = record
    return latest


def write_manual_review(
    evidence_path: Path,
    labels_by_id: dict[str, dict],
    output_path: Path,
    *,
    high_score_sample: int,
    sample_seed: int,
) -> None:
    latest: dict[str, dict] = {}
    for record in read_jsonl(evidence_path):
        if not record.get("error") and isinstance(record.get("parsed_response"), dict):
            latest[str(record["label_id"])] = record
    disputed = [record for record in latest.values() if record.get("category") == "L3"]
    high = [record for record in latest.values() if record.get("category") in {"L1", "L2"}]
    rng = random.Random(sample_seed)
    sample = rng.sample(high, min(high_score_sample, len(high)))
    review = []
    for record in disputed + sample:
        label_id = str(record["label_id"])
        review.append(
            {
                "label": labels_by_id[label_id],
                "judge": record["parsed_response"],
                "category": record["category"],
                "manual_decision": "",
                "manual_notes": "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in review),
        encoding="utf-8",
    )
    temporary.replace(output_path)


def main() -> int:
    args = parse_args()
    labels = read_jsonl(args.labels)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be a positive integer")
        labels = labels[: args.limit]
    stage1 = successful_stage1(args.stage1_evidence)
    missing = [str(label["label_id"]) for label in labels if str(label["label_id"]) not in stage1]
    if missing:
        raise SystemExit(f"missing successful Stage 1 evidence for {len(missing)} labels")

    run_dir = args.run_dir or Path("runtime") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-stage2"
    )
    evidence_path = run_dir / "stage2.evidence.jsonl"
    report_path = run_dir / "stage2.report.json"
    completed = load_completed_ids(evidence_path)
    client = DSClient(
        configured_endpoints(args.endpoints) or list(DEFAULT_ENDPOINTS),
        args.model,
        timeout=args.timeout,
        retries=args.retries,
    )

    for index, label in enumerate(labels, start=1):
        label_id = str(label["label_id"])
        if label_id in completed:
            continue
        prompt = build_alignment_prompt(label, stage1[label_id]["parsed_response"])
        record = {
            "stage": "label_alignment_judge",
            "label_id": label_id,
            "label_name": label["label_name"],
            "prompt": prompt,
            "model": args.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "category": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
            "error": None,
        }
        try:
            response = client.chat(
                [
                    {"role": "system", "content": "你是严谨的高中生物知识点体系审核专家。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=args.max_tokens,
            )
            parsed = validate_alignment_result(parse_json_content(response.content))
            record.update(
                {
                    "raw_response": response.content,
                    "parsed_response": parsed,
                    "category": classify_alignment(parsed),
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
            f"{'ERROR' if record['error'] else record['category']}",
            flush=True,
        )

    labels_by_id = {str(label["label_id"]): label for label in labels}
    write_manual_review(
        evidence_path,
        labels_by_id,
        run_dir / "manual_review.jsonl",
        high_score_sample=args.high_score_sample,
        sample_seed=args.sample_seed,
    )
    report = summarize_evidence(
        evidence_path, (str(item["label_id"]) for item in labels)
    )
    category_counts = {"L1": 0, "L2": 0, "L3": 0}
    for record in read_jsonl(evidence_path):
        category = record.get("category")
        if category in category_counts:
            category_counts[category] += 1
    report["category_counts"] = category_counts
    write_json_atomic(report_path, report)
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["processed"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

