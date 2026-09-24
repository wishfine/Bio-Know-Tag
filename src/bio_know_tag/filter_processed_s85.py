"""Reuse an already-cleaned merged corpus while restoring exact s85 membership."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from bio_know_tag.questions import _base_question


def _update_ids(path: str | Path, subject: str) -> set[str]:
    ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("subject") == subject:
                question_id = str(row.get("question_id") or "")
                if not question_id:
                    raise ValueError("update row missing question_id")
                ids.add(question_id)
    return ids


def _s85_index(path: str | Path, update_ids: set[str]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    parents: dict[str, str] = {}
    replacements: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"s85 line {line_number} must be an object")
            question_id = str(row.get("question_id") or "")
            if not question_id or question_id in parents:
                raise ValueError(f"missing or duplicate s85 question_id at line {line_number}")
            parent_id = str(row.get("parent_id") or question_id)
            parents[question_id] = parent_id
            if question_id in update_ids:
                replacements[question_id] = _base_question(row, question_id, parent_id)
    if not parents:
        raise ValueError("s85 contains no questions")
    return parents, replacements


def filter_processed_s85(
    processed_path: str | Path,
    dedup_raw_path: str | Path,
    updates_path: str | Path,
    output_dir: str | Path,
    *,
    subject: str = "生物",
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Filter merged grouped output by s85 ID, restoring overlaid s85 rows only.

    The original processed file remains untouched. An absent s85 parent is kept
    only as an empty wrapper around retained children, ready for source audit.
    """
    update_ids = _update_ids(updates_path, subject)
    s85_parents, replacements = _s85_index(dedup_raw_path, update_ids)
    remaining = set(s85_parents)
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix=f".{output.name}.filter-", dir=output.parent))
    counts: Counter[str] = Counter()
    try:
        with (
            Path(processed_path).open(encoding="utf-8") as source,
            (work / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as destination,
        ):
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                parent = json.loads(line)
                if not isinstance(parent, dict):
                    raise ValueError(f"processed line {line_number} must be an object")
                parent_id = str(parent.get("question_id") or "")
                if not parent_id:
                    raise ValueError(f"processed line {line_number} missing question_id")
                counts["processed_groups_read"] += 1
                children = parent.get("sub_questions") or []
                if not isinstance(children, list):
                    raise ValueError(f"sub_questions is not a list for {parent_id}")
                retained_children = []
                for child in children:
                    if not isinstance(child, dict):
                        raise ValueError(f"invalid child in group {parent_id}")
                    child_id = str(child.get("question_id") or "")
                    if not child_id:
                        raise ValueError(f"child missing question_id in group {parent_id}")
                    if child_id not in s85_parents:
                        counts["removed_non_s85_question_ids"] += 1
                        continue
                    if child_id not in remaining:
                        raise ValueError(f"duplicate processed s85 question_id: {child_id}")
                    if s85_parents[child_id] != parent_id or str(child.get("parent_id") or "") != parent_id:
                        raise ValueError(f"parent relation mismatch for child {child_id}: expected {s85_parents[child_id]}")
                    remaining.remove(child_id)
                    if child_id in replacements:
                        child = replacements[child_id]
                        counts["restored_updated_s85_records"] += 1
                    retained_children.append(child)

                if parent_id in s85_parents:
                    if parent_id not in remaining:
                        raise ValueError(f"duplicate processed s85 question_id: {parent_id}")
                    if s85_parents[parent_id] != parent_id or str(parent.get("parent_id") or "") != parent_id:
                        raise ValueError(f"parent relation mismatch for parent {parent_id}")
                    remaining.remove(parent_id)
                    if parent_id in replacements:
                        parent = replacements[parent_id]
                        counts["restored_updated_s85_records"] += 1
                    else:
                        parent = dict(parent)
                    parent["sub_questions"] = retained_children
                elif retained_children:
                    # The merged parent may itself be a newly added update row.
                    # Its content must not leak into the s85-only corpus.
                    parent = {
                        "question_id": parent_id,
                        "parent_id": parent_id,
                        "stem": "",
                        "options": "",
                        "answer": "",
                        "analysis": "",
                        "sub_questions": retained_children,
                    }
                    counts["orphan_context_wrappers"] += 1
                    counts["removed_non_s85_question_ids"] += int(
                        parent_id in update_ids
                    )
                else:
                    counts["removed_non_s85_question_ids"] += 1
                    continue
                destination.write(json.dumps(parent, ensure_ascii=False, sort_keys=True) + "\n")
                counts["output_groups"] += 1
                if progress_every and counts["processed_groups_read"] % progress_every == 0:
                    print(f"s85 filter: groups={counts['processed_groups_read']} remaining={len(remaining)}", flush=True)
        if remaining:
            raise ValueError(f"s85 question IDs absent from processed input: {sorted(remaining)[:5]}")
        report = {
            "s85_question_ids": len(s85_parents),
            "processed_groups_read": counts["processed_groups_read"],
            "output_groups": counts["output_groups"],
            "restored_updated_s85_records": counts["restored_updated_s85_records"],
            "removed_non_s85_question_ids": counts["removed_non_s85_question_ids"],
            "orphan_context_wrappers": counts["orphan_context_wrappers"],
            "content_deduplication_applied": False,
            "source_processed_path": str(processed_path),
            "source_s85_path": str(dedup_raw_path),
        }
        (work / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise ValueError(f"output directory appeared during filter: {output}")
        work.replace(output)
        return report
    finally:
        if work.exists():
            shutil.rmtree(work)
