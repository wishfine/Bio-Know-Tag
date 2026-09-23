"""Build one retrieval stream containing standalone, child, and parent material units."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.label_units import IMAGE_REFERENCE_RE


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} must be a JSON object")
            yield row


def _write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_status(unit: dict[str, Any]) -> str:
    stem = str(unit.get("stem") or "").strip()
    options = str(unit.get("options") or "").strip()
    analysis = str(unit.get("analysis") or "").strip()
    if stem:
        return "stem_present"
    if options or analysis:
        return "stem_missing_other_text_present"
    return "no_current_question_text"


def _parent_material_unit(parent: dict[str, Any], child_ids: list[str]) -> dict[str, Any]:
    parent_id = str(parent["question_id"])
    stem = str(parent.get("parent_stem") or "").strip()
    options = str(parent.get("options") or "").strip()
    analysis = str(parent.get("analysis") or "").strip()
    return {
        "question_id": parent_id,
        "parent_id": parent_id,
        "unit_type": "composite_parent_extra",
        "parent_stem": "",
        "stem": stem,
        "options": options,
        "answer_text": str(parent.get("answer_text") or "").strip(),
        "analysis": analysis,
        "child_question_ids": child_ids,
        "legacy_knw_ids": parent.get("legacy_knw_ids") or [],
        "flags": {
            "parent_context_missing": False,
            "image_context_missing": bool(IMAGE_REFERENCE_RE.search(" ".join((stem, options, analysis)))),
        },
    }


def build_unified_retrieval_units(
    units_path: str | Path,
    parent_plans_path: str | Path,
    output_dir: str | Path,
    *,
    keep_question_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Retain canonical child units and append parent-material units in one stream.

    Parent plans are retained when at least one of their children survives. Their
    child ID list is filtered to the same canonical subset. Only text-ineligible
    units are excluded from retrieval, with an audit record instead of deletion.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    units_tmp = output / ".retrieval_units.jsonl.tmp"
    parents_tmp = output / ".parent_aggregation.jsonl.tmp"
    audit_tmp = output / ".content_review.jsonl.tmp"
    excluded_tmp = output / ".text_ineligible.jsonl.tmp"
    counters: Counter[str] = Counter()
    kept_ids: set[str] = set()
    emitted_ids: set[str] = set()

    with (
        units_tmp.open("w", encoding="utf-8") as units_out,
        parents_tmp.open("w", encoding="utf-8") as parents_out,
        audit_tmp.open("w", encoding="utf-8") as audit_out,
        excluded_tmp.open("w", encoding="utf-8") as excluded_out,
    ):
        for unit in _read_jsonl(units_path):
            counters["source_label_units"] += 1
            question_id = str(unit.get("question_id") or "")
            if not question_id:
                raise ValueError("label unit missing question_id")
            if keep_question_ids is not None and question_id not in keep_question_ids:
                counters["deduplicated_units_removed"] += 1
                continue
            if question_id in kept_ids:
                raise ValueError(f"duplicate kept unit: {question_id}")
            kept_ids.add(question_id)
            status = _content_status(unit)
            counters[f"content_status_{status}"] += 1
            if status != "stem_present":
                _write_jsonl(audit_out, {
                    "question_id": question_id,
                    "parent_id": unit.get("parent_id"),
                    "unit_type": unit.get("unit_type"),
                    "content_status": status,
                    "has_parent_stem": bool(str(unit.get("parent_stem") or "").strip()),
                    "has_answer": bool(str(unit.get("answer_text") or "").strip()),
                    "image_context_missing": bool((unit.get("flags") or {}).get("image_context_missing")),
                })
            if status == "no_current_question_text":
                _write_jsonl(excluded_out, unit)
                counters["text_ineligible_label_units"] += 1
                continue
            _write_jsonl(units_out, unit)
            emitted_ids.add(question_id)
            counters["retrieval_label_units"] += 1
            counters[f"retrieval_{unit.get('unit_type') or 'unknown'}"] += 1

        if keep_question_ids is not None:
            missing_ids = keep_question_ids - kept_ids
            if missing_ids:
                raise ValueError(f"keep IDs absent from label units: {sorted(missing_ids)[:5]}")

        for parent in _read_jsonl(parent_plans_path):
            counters["source_parent_plans"] += 1
            parent_id = str(parent.get("question_id") or "")
            if not parent_id:
                raise ValueError("parent plan missing question_id")
            child_ids = [str(value) for value in (parent.get("child_question_ids") or [])]
            retained_children = list(dict.fromkeys(child_id for child_id in child_ids if child_id in kept_ids))
            if not retained_children:
                counters["parents_without_kept_children"] += 1
                continue
            filtered_parent = dict(parent)
            filtered_parent["child_question_ids"] = retained_children
            _write_jsonl(parents_out, filtered_parent)
            counters["retained_parent_plans"] += 1
            parent_unit = _parent_material_unit(parent, retained_children)
            status = _content_status(parent_unit)
            if status == "no_current_question_text":
                counters["parents_without_text_material"] += 1
                continue
            if parent_id in emitted_ids:
                raise ValueError(f"parent ID collides with retrieval unit: {parent_id}")
            if status != "stem_present":
                _write_jsonl(audit_out, {
                    "question_id": parent_id,
                    "parent_id": parent_id,
                    "unit_type": "composite_parent_extra",
                    "content_status": status,
                    "has_parent_stem": False,
                    "has_answer": bool(parent_unit["answer_text"]),
                    "image_context_missing": parent_unit["flags"]["image_context_missing"],
                })
            _write_jsonl(units_out, parent_unit)
            emitted_ids.add(parent_id)
            counters["retrieval_parent_extra_units"] += 1

    units_tmp.replace(output / "retrieval_units.jsonl")
    parents_tmp.replace(output / "parent_aggregation.jsonl")
    audit_tmp.replace(output / "content_review.jsonl")
    excluded_tmp.replace(output / "text_ineligible.jsonl")
    report = {
        **dict(counters),
        "source_label_units": counters["source_label_units"],
        "retrieval_label_units": counters["retrieval_label_units"],
        "retrieval_parent_extra_units": counters["retrieval_parent_extra_units"],
        "text_ineligible_label_units": counters["text_ineligible_label_units"],
        "deduplicated_units_removed": counters["deduplicated_units_removed"],
        "retained_parent_plans": counters["retained_parent_plans"],
        "parents_without_text_material": counters["parents_without_text_material"],
        "retrieval_units": counters["retrieval_label_units"] + counters["retrieval_parent_extra_units"],
        "input_sha256": {
            "label_units": _sha256(units_path),
            "parent_aggregation": _sha256(parent_plans_path),
        },
        "dedup_filter_applied": keep_question_ids is not None,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
