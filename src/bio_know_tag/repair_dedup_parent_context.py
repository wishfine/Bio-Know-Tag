"""Reconnect deduplicated children to source parent context without adding questions."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Any


RECOVERABLE_STATUSES = {"found_with_text_stem", "found_with_other_text"}
SOURCE_PARENT_PRESENT_STATUSES = RECOVERABLE_STATUSES | {
    "found_image_only_stem", "found_without_text"
}


def _load_source_audit(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"source audit line {number} must be an object")
            parent_id = str(row.get("parent_id") or "")
            if not parent_id or parent_id in records:
                raise ValueError(f"missing or duplicate parent_id in source audit line {number}")
            status = str(row.get("status") or "")
            if status in RECOVERABLE_STATUSES:
                if not any(str(row.get(key) or "").strip() for key in ("parent_stem", "options", "analysis")):
                    raise ValueError(f"recoverable parent has no text: {parent_id}")
            elif status not in {"missing_from_original", "found_image_only_stem", "found_without_text", "found_nonself_parent_row"}:
                raise ValueError(f"unknown source status for parent {parent_id}: {status}")
            child_count = int(row.get("child_count") or 0)
            if child_count < 1:
                raise ValueError(f"invalid child count for parent {parent_id}")
            records[parent_id] = row
    if not records:
        raise ValueError("source audit has no parent rows")
    return records


def _load_s85_question_ids(path: str | Path) -> dict[str, str]:
    ids: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"s85 line {number} must be an object")
            question_id = str(row.get("question_id") or "")
            if not question_id or question_id in ids:
                raise ValueError(f"missing or duplicate question_id in s85 line {number}")
            ids[question_id] = str(row.get("parent_id") or question_id)
    if not ids:
        raise ValueError("s85 file contains no questions")
    return ids


def repair_dedup_parent_context(
    processed_path: str | Path,
    source_audit_path: str | Path,
    dedup_raw_path: str | Path,
    output_dir: str | Path,
    *,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Write a new grouped-question stream with orphan context restored or dropped.

    Existing s85 question rows are not changed or content-deduplicated. Recovered
    parent IDs remain context-only wrappers around their retained child rows.
    """
    source = _load_source_audit(source_audit_path)
    remaining_s85_ids = _load_s85_question_ids(dedup_raw_path)
    s85_question_count = len(remaining_s85_ids)
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix=f".{output.name}.repair-", dir=output.parent))
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    try:
        with (
            Path(processed_path).open(encoding="utf-8") as input_handle,
            (work / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as questions,
            (work / "dropped_orphan_groups.jsonl").open("w", encoding="utf-8", newline="\n") as dropped,
            (work / "recovered_context_parents.jsonl").open("w", encoding="utf-8", newline="\n") as recovered,
        ):
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                parent = json.loads(line)
                if not isinstance(parent, dict):
                    raise ValueError(f"processed line {line_number} must be an object")
                parent_id = str(parent.get("question_id") or "")
                if not parent_id:
                    raise ValueError(f"processed line {line_number} missing question_id")
                counts["processed_parent_records"] += 1
                audit = source.get(parent_id)
                if audit is None:
                    if parent_id not in remaining_s85_ids:
                        raise ValueError(f"processed question ID not in s85 or repeated: {parent_id}")
                    if remaining_s85_ids[parent_id] != parent_id:
                        raise ValueError(f"parent relation mismatch for processed parent {parent_id}")
                    remaining_s85_ids.pop(parent_id)
                elif parent_id in remaining_s85_ids:
                    raise ValueError(f"orphan parent unexpectedly exists in s85: {parent_id}")
                children = parent.get("sub_questions") or []
                if not isinstance(children, list):
                    raise ValueError(f"sub_questions must be a list: {parent_id}")
                for child in children:
                    if not isinstance(child, dict):
                        raise ValueError(f"child must be an object: {parent_id}")
                    child_id = str(child.get("question_id") or "")
                    if child_id not in remaining_s85_ids:
                        raise ValueError(f"processed question ID not in s85 or repeated: {child_id}")
                    if str(child.get("parent_id") or "") != parent_id or remaining_s85_ids[child_id] != parent_id:
                        raise ValueError(f"parent relation mismatch for child {child_id}: expected {parent_id}")
                    remaining_s85_ids.pop(child_id)
                if audit is None:
                    questions.write(line if line.endswith("\n") else line + "\n")
                    counts["unchanged_parent_records"] += 1
                    counts["output_parent_records"] += 1
                    continue
                seen.add(parent_id)
                if len(children) != int(audit["child_count"]):
                    raise ValueError(f"child count mismatch for parent {parent_id}")
                child_ids = [str(child.get("question_id") or "") for child in children]
                if not all(child_ids) or len(child_ids) != len(set(child_ids)):
                    raise ValueError(f"missing or duplicate child ID for parent {parent_id}")
                if audit["status"] in SOURCE_PARENT_PRESENT_STATUSES:
                    repaired = dict(parent)
                    stem = str(audit.get("parent_stem") or "")
                    parent_text_missing = not bool(stem.strip())
                    repaired.update({
                        "stem": stem,
                        "options": str(audit.get("options") or ""),
                        "answer": audit.get("answer") or "",
                        "analysis": str(audit.get("analysis") or ""),
                        "knw_ids": [str(value) for value in (audit.get("knw_ids") or []) if value],
                        "context_only_recovered_parent": True,
                        "recovered_parent_has_image": bool(
                            audit.get("raw_material_has_image", audit.get("raw_stem_has_image"))
                        ),
                        "recovered_parent_text_missing": parent_text_missing,
                    })
                    questions.write(json.dumps(repaired, ensure_ascii=False, sort_keys=True) + "\n")
                    recovered.write(json.dumps({
                        "parent_id": parent_id,
                        "child_question_ids": child_ids,
                        "context_only": True,
                        "text_available": not parent_text_missing,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                    if parent_text_missing:
                        counts["retained_source_parents_without_text"] += 1
                        counts["children_with_unavailable_parent_text"] += len(children)
                    else:
                        counts["recovered_context_parents"] += 1
                        counts["children_with_recovered_context"] += len(children)
                    counts["output_parent_records"] += 1
                else:
                    dropped.write(json.dumps({
                        "parent_id": parent_id,
                        "child_question_ids": child_ids,
                        "reason": audit["status"],
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["dropped_orphan_parent_groups"] += 1
                    counts["dropped_orphan_children"] += len(children)
                if progress_every and counts["processed_parent_records"] % progress_every == 0:
                    print(
                        f"repair: processed={counts['processed_parent_records']} "
                        f"recovered={counts['recovered_context_parents']} "
                        f"dropped_children={counts['dropped_orphan_children']}",
                        flush=True,
                    )
        missing_audit_ids = set(source) - seen
        if missing_audit_ids:
            raise ValueError(f"source audit parent IDs absent from processed questions: {sorted(missing_audit_ids)[:5]}")
        if remaining_s85_ids:
            raise ValueError(f"s85 question IDs absent from processed questions: {sorted(remaining_s85_ids)[:5]}")
        (work / "orphan_parents.jsonl").write_text("", encoding="utf-8")
        report = {
            "processed_parent_records": counts["processed_parent_records"],
            "output_parent_records": counts["output_parent_records"],
            "unchanged_parent_records": counts["unchanged_parent_records"],
            "recovered_context_parents": counts["recovered_context_parents"],
            "children_with_recovered_context": counts["children_with_recovered_context"],
            "retained_source_parents_without_text": counts["retained_source_parents_without_text"],
            "children_with_unavailable_parent_text": counts["children_with_unavailable_parent_text"],
            "dropped_orphan_parent_groups": counts["dropped_orphan_parent_groups"],
            "dropped_orphan_children": counts["dropped_orphan_children"],
            "independent_parent_records_added": 0,
            "content_deduplication_applied": False,
            "s85_question_ids": s85_question_count,
            "output_s85_question_ids": s85_question_count - counts["dropped_orphan_children"],
            "source_dedup_raw_path": str(dedup_raw_path),
            "source_processed_path": str(processed_path),
            "source_audit_path": str(source_audit_path),
        }
        (work / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise ValueError(f"output directory appeared during repair: {output}")
        work.replace(output)
        return report
    finally:
        if work.exists():
            shutil.rmtree(work)
