"""Build a deterministic representative-plus-stress DS adjudication sample."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rank(question_id: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _retrieval_top1_disagrees(candidate_row: dict[str, Any]) -> bool:
    candidates = candidate_row.get("candidates") or []
    sparse = min(
        (
            (int(candidate["sparse_rank"]), str(candidate["label_id"]))
            for candidate in candidates
            if candidate.get("sparse_rank") is not None
        ),
        default=None,
    )
    dense = min(
        (
            (int(candidate["dense_rank"]), str(candidate["label_id"]))
            for candidate in candidates
            if candidate.get("dense_rank") is not None
        ),
        default=None,
    )
    return bool(sparse and dense and sparse[1] != dense[1])


def _top_module(candidate_row: dict[str, Any]) -> str:
    candidates = candidate_row.get("candidates") or []
    if not candidates:
        return "missing"
    path = str(candidates[0].get("label_path") or "").replace("->", "@")
    parts = [part.strip() for part in path.split("@") if part.strip()]
    return parts[1] if len(parts) > 1 else "missing"


def _strata(unit: dict[str, Any], candidate_row: dict[str, Any]) -> list[str]:
    metadata = unit.get("metadata") or {}
    flags = unit.get("flags") or {}
    return [
        f"unit_type={unit.get('unit_type') or 'missing'}",
        f"difficulty={metadata.get('difficulty') or 'missing'}",
        f"structure_type={metadata.get('structure_type') or 'missing'}",
        f"top_module={_top_module(candidate_row)}",
        f"retrieval_top1_disagree={str(_retrieval_top1_disagrees(candidate_row)).lower()}",
        f"image_context_missing={str(bool(flags.get('image_context_missing'))).lower()}",
        f"empty_answer={str(not bool(str(unit.get('answer_text') or '').strip())).lower()}",
        f"empty_analysis={str(not bool(str(unit.get('analysis') or '').strip())).lower()}",
    ]


def build_adjudication_audit_sample(
    units_path: str | Path,
    candidates_path: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int = 300,
    representative_size: int = 200,
    seed: str = "adjudication-audit-v1",
) -> dict[str, Any]:
    """Select a uniform metric split plus a coverage-oriented stress split."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if representative_size < 0 or representative_size > sample_size:
        raise ValueError("representative_size must be between 0 and sample_size")
    units = _read_jsonl(units_path)
    candidates = _read_jsonl(candidates_path)
    if sample_size > len(units):
        raise ValueError("sample_size exceeds available units")
    units_by_id = {str(row.get("question_id") or ""): row for row in units}
    candidates_by_id = {
        str(row.get("question_id") or ""): row for row in candidates
    }
    if "" in units_by_id or len(units_by_id) != len(units):
        raise ValueError("units must have unique non-empty question_id values")
    if set(units_by_id) != set(candidates_by_id):
        raise ValueError("units and candidates must contain identical question IDs")

    ordered_ids = sorted(
        units_by_id,
        key=lambda question_id: (_rank(question_id, f"{seed}:representative"), question_id),
    )
    representative_ids = ordered_ids[:representative_size]
    selected = set(representative_ids)
    stress_size = sample_size - representative_size
    strata_by_id = {
        question_id: _strata(units_by_id[question_id], candidates_by_id[question_id])
        for question_id in units_by_id
    }
    population_counts: Counter[str] = Counter(
        stratum for values in strata_by_id.values() for stratum in values
    )
    covered: Counter[str] = Counter()
    remaining = set(units_by_id) - selected
    stress_ids: list[str] = []
    special_prefixes = {
        "unit_type=orphan_sub_question",
        "retrieval_top1_disagree=true",
        "image_context_missing=true",
        "empty_answer=true",
        "empty_analysis=true",
    }
    for _ in range(stress_size):
        if not remaining:
            break

        def priority(question_id: str) -> tuple[float, int, str]:
            score = 0.0
            for stratum in strata_by_id[question_id]:
                rarity = 1.0 / population_counts[stratum]
                weight = 3.0 if stratum in special_prefixes else 1.0
                score += weight * rarity / (1 + covered[stratum])
            return (-score, _rank(question_id, f"{seed}:stress"), question_id)

        question_id = min(remaining, key=priority)
        remaining.remove(question_id)
        selected.add(question_id)
        stress_ids.append(question_id)
        covered.update(strata_by_id[question_id])

    selected_rows: list[tuple[str, str]] = [
        (question_id, "representative") for question_id in representative_ids
    ] + [(question_id, "stress") for question_id in stress_ids]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_rows = []
    candidate_rows = []
    manifest_rows = []
    for question_id, split in selected_rows:
        strata = strata_by_id[question_id]
        unit_rows.append(
            {
                **units_by_id[question_id],
                "audit_split": split,
                "audit_strata": strata,
            }
        )
        candidate_rows.append(
            {
                **candidates_by_id[question_id],
                "audit_split": split,
                "audit_strata": strata,
            }
        )
        manifest_rows.append(
            {
                "question_id": question_id,
                "audit_split": split,
                "audit_strata": strata,
            }
        )
    _write_jsonl_atomic(output / "audit_units.jsonl", unit_rows)
    _write_jsonl_atomic(output / "audit_candidates.jsonl", candidate_rows)
    _write_jsonl_atomic(output / "audit_manifest.jsonl", manifest_rows)

    split_counts = Counter(row["audit_split"] for row in unit_rows)
    unit_type_counts = Counter(
        str(row.get("unit_type") or "missing") for row in unit_rows
    )
    sample_strata = Counter(
        stratum for row in unit_rows for stratum in row["audit_strata"]
    )
    report = {
        "input_units": len(units),
        "sample_size": len(unit_rows),
        "representative_size": len(representative_ids),
        "stress_size": len(stress_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "unit_type_counts": dict(sorted(unit_type_counts.items())),
        "sample_strata": dict(sorted(sample_strata.items())),
        "seed": seed,
        "sampling_version": "representative-plus-stress-v1",
        "legacy_fields_used": False,
    }
    _write_json_atomic(output / "report.json", report)
    print(
        "audit sample explains recall/precision on a uniform split and failure modes on a stress split",
        flush=True,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report
