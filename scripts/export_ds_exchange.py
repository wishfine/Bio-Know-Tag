#!/usr/bin/env python3
"""Export the reconstructed request and recorded response for one DS exchange."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bio_know_tag.adjudication import build_adjudication_prompt


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def by_question(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("question_id") or ""): row
        for row in read_rows(path)
        if row.get("question_id")
    }


def labels_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("label_id")): row
        for row in read_rows(path)
        if row.get("label_id")
    }


def manifest_for(evidence_path: Path) -> dict[str, Any]:
    path = evidence_path.parent / "run_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_one(
    *,
    name: str,
    question_id: str,
    units: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    candidates_path: Path,
    evidence_path: Path,
) -> dict[str, Any] | None:
    evidence = by_question(evidence_path)
    if question_id not in evidence:
        return None
    if question_id not in units:
        raise SystemExit(f"{question_id} 不在 units 文件中")

    candidate_rows = by_question(candidates_path)
    candidate_row = candidate_rows[question_id]
    candidate_list = candidate_row.get("candidates") or []

    # Older runs can contain a candidate card whose Label was later removed
    # from the current catalog. Keep its embedded card as a fallback so the
    # exchange can still be inspected; the hash check exposes any mismatch.
    labels_for_prompt = dict(labels)
    for candidate in candidate_list:
        label_id = str(candidate.get("label_id") or "")
        if label_id and label_id not in labels_for_prompt:
            labels_for_prompt[label_id] = candidate

    prompt, code_map = build_adjudication_prompt(
        units[question_id], candidate_list, labels_for_prompt
    )
    evidence_row = evidence[question_id]
    recorded_sha = evidence_row.get("prompt_sha256")
    reconstructed_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    manifest = manifest_for(evidence_path)

    return {
        "strategy": name,
        "run_dir": str(evidence_path.parent),
        "question_id": question_id,
        "request": {
            "model": evidence_row.get("model") or manifest.get("model"),
            "endpoint": evidence_row.get("endpoint"),
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的高中生物知识点判标器，只输出JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            "candidate_code_map": code_map,
            "candidate_count": len(candidate_list),
            "prompt_version": evidence_row.get("prompt_version"),
            "prompt_sha256_recorded": recorded_sha,
            "prompt_sha256_reconstructed": reconstructed_sha,
            "prompt_sha256_match": recorded_sha == reconstructed_sha,
            "max_tokens": manifest.get("max_tokens"),
        },
        "response": {
            key: evidence_row.get(key)
            for key in (
                "raw_response",
                "parsed_response",
                "reasoning",
                "usage",
                "attempts",
                "latency_seconds",
                "retry_errors",
                "error",
                "created_at",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--top25-evidence", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--legacy-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    units = by_question(args.units)
    labels = labels_by_id(args.labels)
    exchanges = {}
    for name, candidates, evidence in (
        ("top25", args.top25_candidates, args.top25_evidence),
        ("top25_plus_legacy", args.legacy_candidates, args.legacy_evidence),
    ):
        value = export_one(
            name=name,
            question_id=args.question_id,
            units=units,
            labels=labels,
            candidates_path=candidates,
            evidence_path=evidence,
        )
        if value is not None:
            exchanges[name] = value

    result = {
        "question_id": args.question_id,
        "strategies_found": list(exchanges),
        "exchanges": exchanges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "strategies_found": list(exchanges)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
