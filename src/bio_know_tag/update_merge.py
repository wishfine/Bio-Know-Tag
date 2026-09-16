"""Safely overlay subject-specific question updates onto a raw JSONL baseline."""

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


def _ids(row: dict[str, Any]) -> list[str]:
    return [str(value) for value in (row.get("knw_ids") or [])]


def merge_question_updates(
    base_path: str | Path,
    updates_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    subject: str,
) -> dict[str, Any]:
    """Overlay the last update for each question ID and append genuinely new rows.

    The baseline is never changed. Update rows belonging to other subjects are ignored.
    Output is replaced atomically after the complete merged file has been written.
    """
    updates: dict[str, dict[str, Any]] = {}
    update_rows_scanned = subject_update_rows = update_errors = duplicate_updates = 0
    with Path(updates_path).open(encoding="utf-8") as handle:
        for line in handle:
            update_rows_scanned += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("update row must be an object")
                if str(row.get("subject") or "") != subject:
                    continue
                question_id = str(row.get("question_id") or "").strip()
                if not question_id:
                    raise ValueError("question_id is required")
                subject_update_rows += 1
                if question_id in updates:
                    duplicate_updates += 1
                updates[question_id] = row
            except (json.JSONDecodeError, TypeError, ValueError):
                update_errors += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    seen: set[str] = set()
    base_rows = base_errors = existing_updated = label_changed = output_rows = 0
    with Path(base_path).open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            base_rows += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("base row must be an object")
                question_id = str(row.get("question_id") or "").strip()
                if not question_id:
                    raise ValueError("question_id is required")
            except (json.JSONDecodeError, TypeError, ValueError):
                base_errors += 1
                continue
            replacement = updates.get(question_id)
            if replacement is not None:
                existing_updated += 1
                label_changed += _ids(row) != _ids(replacement)
                row = replacement
                seen.add(question_id)
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output_rows += 1

        for question_id, row in updates.items():
            if question_id in seen:
                continue
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output_rows += 1

    temporary.replace(output)
    new_rows_added = len(updates) - len(seen)
    report = {
        "subject": subject,
        "base_rows": base_rows,
        "base_errors": base_errors,
        "update_rows_scanned": update_rows_scanned,
        "subject_update_rows": subject_update_rows,
        "unique_subject_update_questions": len(updates),
        "duplicate_update_question_ids": duplicate_updates,
        "update_errors": update_errors,
        "existing_rows_updated": existing_updated,
        "label_changed_existing_rows": label_changed,
        "new_rows_added": new_rows_added,
        "output_rows": output_rows,
    }
    _write_json_atomic(Path(report_path), report)
    return report

