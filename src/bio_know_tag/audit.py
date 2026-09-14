"""Read-only audits for preprocessed question data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_orphan_parents(
    raw_path: str | Path,
    processed_path: str | Path,
    evidence_path: str | Path,
    report_path: str | Path,
    *,
    progress_every: int = 100_000,
) -> dict[str, int]:
    """Find referenced parent IDs absent from raw rows and inspect their containers."""
    raw_input = 0
    raw_error = 0
    question_ids: set[str] = set()
    referenced_parent_ids: set[str] = set()

    with Path(raw_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_input += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("raw row must be an object")
                question_id = str(row.get("question_id") or "").strip()
                if not question_id:
                    raise ValueError("question_id is required")
                parent_id = str(row.get("parent_id") or question_id).strip()
                question_ids.add(question_id)
                if parent_id != question_id:
                    referenced_parent_ids.add(parent_id)
            except (json.JSONDecodeError, TypeError, ValueError):
                raw_error += 1
            if progress_every and raw_input % progress_every == 0:
                print(
                    f"raw scan: {raw_input} rows, error={raw_error}",
                    flush=True,
                )

    orphan_ids = referenced_parent_ids - question_ids
    evidence_output = Path(evidence_path)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_evidence = evidence_output.with_name(f".{evidence_output.name}.tmp")
    processed_input = 0
    processed_error = 0
    found_ids: set[str] = set()
    with_parent_stem = 0
    without_parent_stem = 0
    orphan_child_count = 0
    children_without_stem = 0

    with Path(processed_path).open("r", encoding="utf-8") as source, temporary_evidence.open(
        "w", encoding="utf-8", newline="\n"
    ) as evidence:
        for line in source:
            processed_input += 1
            try:
                parent = json.loads(line)
                if not isinstance(parent, dict):
                    raise ValueError("processed row must be an object")
                parent_id = str(parent.get("question_id") or "").strip()
                if parent_id not in orphan_ids:
                    continue
                children = parent.get("sub_questions")
                if not isinstance(children, list):
                    raise ValueError("sub_questions must be a list")
                parent_stem_present = bool(str(parent.get("stem") or "").strip())
                child_missing = sum(
                    not isinstance(child, dict)
                    or not str(child.get("stem") or "").strip()
                    for child in children
                )
                found_ids.add(parent_id)
                with_parent_stem += int(parent_stem_present)
                without_parent_stem += int(not parent_stem_present)
                orphan_child_count += len(children)
                children_without_stem += child_missing
                evidence.write(
                    json.dumps(
                        {
                            "orphan_parent_id": parent_id,
                            "quality": {
                                "parent_stem_present": parent_stem_present,
                                "parent_analysis_present": bool(
                                    str(parent.get("analysis") or "").strip()
                                ),
                                "parent_answer_present": bool(parent.get("answer")),
                                "sub_question_count": len(children),
                                "children_without_stem": child_missing,
                            },
                            "parent": parent,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                evidence.write("\n")
            except (json.JSONDecodeError, TypeError, ValueError):
                processed_error += 1
            if progress_every and processed_input % progress_every == 0:
                print(
                    f"processed scan: {processed_input} rows, "
                    f"orphans_found={len(found_ids)}, error={processed_error}",
                    flush=True,
                )

    temporary_evidence.replace(evidence_output)
    report = {
        "raw_input": raw_input,
        "raw_error": raw_error,
        "unique_question_ids": len(question_ids),
        "referenced_parent_ids": len(referenced_parent_ids),
        "orphan_parent_ids": len(orphan_ids),
        "orphan_records_found": len(found_ids),
        "orphan_records_missing": len(orphan_ids - found_ids),
        "orphan_with_parent_stem": with_parent_stem,
        "orphan_without_parent_stem": without_parent_stem,
        "orphan_child_count": orphan_child_count,
        "orphan_children_without_stem": children_without_stem,
        "processed_input": processed_input,
        "processed_error": processed_error,
    }
    _write_json_atomic(Path(report_path), report)
    return report

