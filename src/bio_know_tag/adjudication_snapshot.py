"""Materialize a paired snapshot from append-only adjudication evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.adjudication import apply_audited_exclusions, load_audited_exclusions


def _read(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _latest_success(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _read(path):
        question_id = str(row.get("question_id") or "")
        if question_id and not row.get("error") and isinstance(row.get("parsed_response"), dict):
            rows[question_id] = row
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize(
    evidence: dict[str, Any],
    candidate_row: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    exclusions: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    question_id = str(evidence["question_id"])
    parsed = evidence["parsed_response"]
    code_map = evidence.get("candidate_code_map") or {}
    evidence_by_code = parsed.get("evidence") or {}
    candidates = {
        str(item["label_id"]): item for item in candidate_row.get("candidates") or []
    }
    selected = []
    for code in parsed.get("selected") or []:
        label_id = str(code_map.get(code) or "")
        if not label_id or label_id not in labels or label_id not in candidates:
            continue
        item = dict(candidates[label_id])
        item["label_name"] = str(labels[label_id].get("label_name") or item.get("label_name") or "")
        item["label_path"] = str(labels[label_id].get("label_path") or item.get("label_path") or "").replace("->", "@")
        item["evidence"] = str(evidence_by_code.get(code) or "")
        selected.append(item)
    selected, removed, exclude_training = apply_audited_exclusions(
        question_id, selected, exclusions
    )
    empty = not selected
    context = bool(parsed.get("context_insufficient"))
    expand = bool(parsed.get("need_expand_recall"))
    return {
        "question_id": question_id,
        "selected_labels": selected,
        "audited_excluded_labels": removed,
        "context_insufficient": context,
        "need_expand_recall": expand,
        "none_of_candidates": empty,
        "needs_review": context or expand or exclude_training,
        "usable_for_training": not (empty or context or expand or exclude_training),
        "reason": str(parsed.get("reason") or ""),
        "model": evidence.get("model"),
        "prompt_version": evidence.get("prompt_version"),
        "retrieval_version": candidate_row.get("retrieval_version"),
    }


def build_paired_adjudication_snapshot(
    units_path: str | Path,
    top25_evidence_path: str | Path,
    legacy_evidence_path: str | Path,
    top25_candidates_path: str | Path,
    legacy_candidates_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    audited_exclusions_path: str | Path | None = None,
) -> dict[str, Any]:
    top_evidence = _latest_success(top25_evidence_path)
    legacy_evidence = _latest_success(legacy_evidence_path)
    common = set(top_evidence) & set(legacy_evidence)
    labels = {str(row["label_id"]): row for row in _read(labels_path)}
    exclusions = (
        load_audited_exclusions(audited_exclusions_path)
        if audited_exclusions_path is not None
        else {}
    )
    top_candidates = {
        str(row["question_id"]): row
        for row in _read(top25_candidates_path)
        if str(row.get("question_id") or "") in common
    }
    legacy_candidates = {
        str(row["question_id"]): row
        for row in _read(legacy_candidates_path)
        if str(row.get("question_id") or "") in common
    }
    units = [row for row in _read(units_path) if str(row.get("question_id") or "") in common]
    ordered_ids = [str(row["question_id"]) for row in units]
    if set(top_candidates) != common or set(legacy_candidates) != common:
        raise ValueError("candidate files are missing common successful question IDs")
    top_predictions = [
        _materialize(top_evidence[qid], top_candidates[qid], labels, exclusions)
        for qid in ordered_ids
    ]
    legacy_predictions = [
        _materialize(legacy_evidence[qid], legacy_candidates[qid], labels, exclusions)
        for qid in ordered_ids
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "units.jsonl", units)
    _write_jsonl(output / "top25_predictions.jsonl", top_predictions)
    _write_jsonl(output / "legacy_predictions.jsonl", legacy_predictions)
    report = {
        "top25_success": len(top_evidence),
        "legacy_success": len(legacy_evidence),
        "common_success": len(common),
        "top25_only_success": len(set(top_evidence) - set(legacy_evidence)),
        "legacy_only_success": len(set(legacy_evidence) - set(top_evidence)),
        "input_sha256": {
            "units": _sha(units_path),
            "top25_evidence": _sha(top25_evidence_path),
            "legacy_evidence": _sha(legacy_evidence_path),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report
