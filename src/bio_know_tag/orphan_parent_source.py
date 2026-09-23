"""Locate orphan parent records in a pre-dedup raw question source."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from bio_know_tag.questions import clean_question_info


def _raw_question_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if value in (None, ""):
        return {}
    raise ValueError("question_info must be an object or object-encoded string")


def audit_orphan_parent_source(
    orphan_audit_path: str | Path,
    original_raw_path: str | Path,
    output_dir: str | Path,
    *,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Audit source presence without changing the deduplicated question set."""
    orphan_counts: dict[str, int] = {}
    with Path(orphan_audit_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"orphan audit line {line_number} must be an object")
            parent_id = str(row.get("orphan_parent_id") or "").strip()
            if not parent_id or parent_id in orphan_counts:
                raise ValueError(f"missing or duplicate orphan_parent_id at line {line_number}")
            quality = row.get("quality") or {}
            child_count = int(quality.get("sub_question_count") or len((row.get("parent") or {}).get("sub_questions") or []))
            if child_count < 1:
                raise ValueError(f"orphan parent {parent_id} has no children")
            orphan_counts[parent_id] = child_count
    if not orphan_counts:
        raise ValueError("orphan audit contains no parent IDs")

    found: dict[str, dict[str, Any]] = {}
    raw_rows = malformed_rows = 0
    with Path(original_raw_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw_rows += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("raw row must be an object")
                question_id = str(row.get("question_id") or "").strip()
                if not question_id:
                    raise ValueError("question_id is required")
            except (json.JSONDecodeError, TypeError, ValueError):
                malformed_rows += 1
                continue
            if question_id not in orphan_counts:
                if progress_every and raw_rows % progress_every == 0:
                    print(f"source scan: {raw_rows} rows, found={len(found)}", flush=True)
                continue
            if question_id in found:
                raise ValueError(f"duplicate target parent question_id in original raw: {question_id}")
            try:
                raw_info = _raw_question_info(row.get("question_info"))
                cleaned = clean_question_info(raw_info)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid question_info for original parent {question_id}: {exc}") from exc
            raw_stem_has_image = "<img" in str(raw_info.get("stem") or "").lower()
            stem = cleaned["stem"]
            if stem:
                status = "found_with_text_stem"
            elif cleaned["options"] or cleaned["analysis"]:
                status = "found_with_other_text"
            elif raw_stem_has_image:
                status = "found_image_only_stem"
            else:
                status = "found_without_text"
            found[question_id] = {
                "parent_id": question_id,
                "child_count": orphan_counts[question_id],
                "status": status,
                "parent_stem": stem,
                "options": cleaned["options"],
                "analysis": cleaned["analysis"],
                "raw_stem_has_image": raw_stem_has_image,
                "original_parent_id": str(row.get("parent_id") or question_id),
                "original_line_number": line_number,
            }
            if progress_every and raw_rows % progress_every == 0:
                print(f"source scan: {raw_rows} rows, found={len(found)}", flush=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    child_counts: Counter[str] = Counter()
    evidence_tmp = output / ".per_parent.jsonl.tmp"
    with evidence_tmp.open("w", encoding="utf-8", newline="\n") as destination:
        for parent_id in sorted(orphan_counts):
            row = found.get(parent_id) or {
                "parent_id": parent_id,
                "child_count": orphan_counts[parent_id],
                "status": "missing_from_original",
                "parent_stem": "",
                "options": "",
                "analysis": "",
                "raw_stem_has_image": False,
                "original_parent_id": None,
                "original_line_number": None,
            }
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[row["status"]] += 1
            child_counts[row["status"]] += row["child_count"]
    evidence_tmp.replace(output / "per_parent.jsonl")
    report = {
        "orphan_parent_ids": len(orphan_counts),
        "orphan_child_count": sum(orphan_counts.values()),
        "found_in_original": len(found),
        "missing_from_original": counts["missing_from_original"],
        "children_with_original_parent": sum(orphan_counts[parent_id] for parent_id in found),
        "children_missing_original_parent": child_counts["missing_from_original"],
        "found_with_text_stem": counts["found_with_text_stem"],
        "found_with_other_text": counts["found_with_other_text"],
        "found_image_only_stem": counts["found_image_only_stem"],
        "found_without_text": counts["found_without_text"],
        "child_counts_by_status": dict(sorted(child_counts.items())),
        "original_raw_rows_scanned": raw_rows,
        "original_raw_malformed_rows": malformed_rows,
        "orphan_audit_path": str(orphan_audit_path),
        "original_raw_path": str(original_raw_path),
        "note": "Source presence does not prove why a parent was absent from the deduplicated file; no records are added back by this audit.",
    }
    report_tmp = output / ".report.json.tmp"
    report_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_tmp.replace(output / "report.json")
    return report
