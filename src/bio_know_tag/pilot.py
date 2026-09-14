"""Build a deterministic Pilot sample without using legacy knowledge IDs."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _rank(key: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class _StableSampler:
    """Keep a deterministic, bounded sample with the lowest hash ranks."""

    def __init__(self, capacity: int):
        self.capacity = max(0, capacity)
        self._heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, key: str, item: dict[str, Any], rank: int) -> None:
        if self.capacity == 0:
            return
        entry = (-rank, key, item)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def items(self) -> list[dict[str, Any]]:
        return [
            item
            for _, _, item in sorted(
                self._heap, key=lambda entry: (-entry[0], entry[1])
            )
        ]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_file(
    path: str | Path,
    *,
    capacity: int,
    seed: str,
    key_field: str,
) -> list[dict[str, Any]]:
    sampler = _StableSampler(capacity)
    for record in _read_jsonl(path):
        key = str(record.get(key_field) or "")
        sampler.add(key, record, _rank(key, seed))
    return sampler.items()


def _stratum_keys(unit: dict[str, Any]) -> list[str]:
    metadata = unit.get("metadata") or {}
    unit_type = str(unit.get("unit_type") or "unknown")
    return [
        f"unit_type={unit_type}",
        f"business_type={metadata.get('business_type') or 'missing'}",
        f"structure_type={metadata.get('structure_type') or 'missing'}",
        f"difficulty={metadata.get('difficulty') or 'missing'}",
        f"unit_type={unit_type}|structure_type={metadata.get('structure_type') or 'missing'}",
    ]


def _without_legacy_fields(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in record.items()
        if not key.startswith("legacy_")
        and key not in {"proposed_route", "route_reason"}
    }
    flags = cleaned.get("flags")
    if isinstance(flags, dict):
        cleaned["flags"] = {
            key: value
            for key, value in flags.items()
            if key != "duplicate_label_conflict"
        }
    return cleaned


def build_pilot_package(
    label_units_path: str | Path,
    parent_aggregation_path: str | Path,
    duplicate_groups_path: str | Path,
    output_dir: str | Path,
    *,
    target_size: int = 2_500,
    parent_groups: int = 200,
    duplicate_groups: int = 100,
    audit_sample_size: int = 100,
    stratum_sample_size: int = 3,
    seed: str = "bio-pilot-no-legacy-v1",
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Create a content/quality-stratified Pilot without reading legacy IDs."""
    if target_size < 1:
        raise ValueError("target_size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    selected_parents = _sample_file(
        parent_aggregation_path,
        capacity=parent_groups,
        seed=f"{seed}:parents",
        key_field="question_id",
    )
    selected_parent_ids = {
        str(parent.get("question_id") or "") for parent in selected_parents
    }
    selected_duplicate_groups = _sample_file(
        duplicate_groups_path,
        capacity=duplicate_groups,
        seed=f"{seed}:duplicates",
        key_field="dedupe_hash",
    )
    duplicate_question_ids: set[str] = set()
    for group in selected_duplicate_groups:
        question_ids = [str(value) for value in group.get("question_ids") or []]
        duplicate_question_ids.update(question_ids[:2])

    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    all_empty_stem: dict[str, dict[str, Any]] = {}
    input_counts: Counter[str] = Counter()
    population_strata: Counter[str] = Counter()
    type_samplers = {
        unit_type: _StableSampler(target_size)
        for unit_type in ("standalone", "sub_question", "orphan_sub_question")
    }
    stratum_samplers: dict[str, _StableSampler] = {}
    orphan_sampler = _StableSampler(audit_sample_size)
    image_samplers: dict[str, _StableSampler] = {}
    empty_answer_sampler = _StableSampler(audit_sample_size)
    empty_analysis_sampler = _StableSampler(audit_sample_size)

    def select(unit: dict[str, Any], reason: str) -> None:
        question_id = str(unit.get("question_id") or "")
        if question_id:
            selected[question_id] = unit
            reasons[question_id].add(reason)

    for index, unit in enumerate(_read_jsonl(label_units_path), 1):
        question_id = str(unit.get("question_id") or "")
        parent_id = str(unit.get("parent_id") or question_id)
        unit_type = str(unit.get("unit_type") or "unknown")
        flags = unit.get("flags") or {}
        rank = _rank(question_id, seed)
        input_counts["units"] += 1
        input_counts[f"unit_type:{unit_type}"] += 1

        if parent_id in selected_parent_ids and unit_type == "sub_question":
            select(unit, "complete_parent_group")
        if question_id in duplicate_question_ids:
            select(unit, "exact_duplicate_sample")
        if not str(unit.get("stem") or "").strip():
            select(unit, "empty_stem")
            all_empty_stem[question_id] = unit

        if unit_type in type_samplers:
            type_samplers[unit_type].add(question_id, unit, rank)
        if unit_type == "orphan_sub_question":
            orphan_sampler.add(question_id, unit, rank)
        if flags.get("image_context_missing"):
            image_samplers.setdefault(
                unit_type, _StableSampler(audit_sample_size)
            ).add(question_id, unit, rank)
        if not str(unit.get("answer_text") or "").strip():
            empty_answer_sampler.add(question_id, unit, rank)
        if not str(unit.get("analysis") or "").strip():
            empty_analysis_sampler.add(question_id, unit, rank)

        for stratum in _stratum_keys(unit):
            population_strata[stratum] += 1
            stratum_samplers.setdefault(
                stratum, _StableSampler(stratum_sample_size)
            ).add(question_id, unit, rank)

        if progress_every and index % progress_every == 0:
            print(
                f"pilot scan: units={index}, mandatory={len(selected)}, strata={len(stratum_samplers)}",
                flush=True,
            )

    for unit in orphan_sampler.items():
        select(unit, "orphan_sub_question")
    for unit_type, sampler in image_samplers.items():
        for unit in sampler.items():
            select(unit, f"image_context:{unit_type}")
    for unit in empty_answer_sampler.items():
        select(unit, "empty_answer_text")
    for unit in empty_analysis_sampler.items():
        select(unit, "empty_analysis")
    for stratum, sampler in sorted(stratum_samplers.items()):
        for unit in sampler.items():
            select(unit, f"stratum:{stratum}")

    desired_type_counts = {
        "standalone": round(target_size * 0.55),
        "sub_question": round(target_size * 0.40),
        "orphan_sub_question": target_size
        - round(target_size * 0.55)
        - round(target_size * 0.40),
    }
    selected_type_counts = Counter(
        str(unit.get("unit_type") or "unknown") for unit in selected.values()
    )
    for unit_type in ("standalone", "sub_question", "orphan_sub_question"):
        for unit in type_samplers[unit_type].items():
            if len(selected) >= target_size:
                break
            if selected_type_counts[unit_type] >= desired_type_counts[unit_type]:
                break
            before = len(selected)
            select(unit, f"type_fill:{unit_type}")
            if len(selected) > before:
                selected_type_counts[unit_type] += 1

    if len(selected) < target_size:
        pooled = []
        for unit_type, sampler in type_samplers.items():
            for unit in sampler.items():
                question_id = str(unit.get("question_id") or "")
                pooled.append(
                    (
                        _rank(question_id, f"{seed}:final-fill"),
                        unit_type,
                        unit,
                    )
                )
        for _, unit_type, unit in sorted(pooled, key=lambda item: (item[0], item[1])):
            if len(selected) >= target_size:
                break
            select(unit, f"type_fill:{unit_type}")

    pilot_rows = []
    for question_id, unit in selected.items():
        row = _without_legacy_fields(unit)
        row["sample_reasons"] = sorted(reasons[question_id])
        pilot_rows.append(row)
    pilot_rows.sort(
        key=lambda unit: (
            unit.get("unit_type", ""),
            unit.get("parent_id", ""),
            unit.get("question_id", ""),
        )
    )
    selected_ids = set(selected)
    pilot_parents = [
        _without_legacy_fields(parent)
        for parent in selected_parents
        if any(
            str(child_id) in selected_ids
            for child_id in parent.get("child_question_ids") or []
        )
    ]
    duplicate_samples = []
    for group in selected_duplicate_groups:
        row = {
            key: value
            for key, value in group.items()
            if key
            in {
                "dedupe_hash",
                "primary_question_id",
                "question_ids",
                "member_count",
            }
        }
        row["sample_units"] = [
            _without_legacy_fields(selected[str(question_id)])
            for question_id in (group.get("question_ids") or [])[:2]
            if str(question_id) in selected
        ]
        duplicate_samples.append(row)
    image_samples = [
        {
            **_without_legacy_fields(unit),
            "sample_reason": f"image_context:{unit_type}",
        }
        for unit_type, sampler in image_samplers.items()
        for unit in sampler.items()
    ]

    _write_jsonl_atomic(output / "pilot_units.jsonl", pilot_rows)
    _write_jsonl_atomic(output / "pilot_parents.jsonl", pilot_parents)
    _write_jsonl_atomic(output / "duplicate_samples.jsonl", duplicate_samples)
    _write_jsonl_atomic(output / "image_context_samples.jsonl", image_samples)
    _write_jsonl_atomic(
        output / "empty_stem_units.jsonl",
        (
            _without_legacy_fields(unit)
            for unit in sorted(
                all_empty_stem.values(), key=lambda unit: unit["question_id"]
            )
        ),
    )

    pilot_type_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    pilot_strata: Counter[str] = Counter()
    for unit in pilot_rows:
        pilot_type_counts[str(unit.get("unit_type") or "unknown")] += 1
        reason_counts.update(unit.get("sample_reasons") or [])
        pilot_strata.update(_stratum_keys(unit))
    report = {
        "target_size": target_size,
        "pilot_units": len(pilot_rows),
        "pilot_parent_groups": len(pilot_parents),
        "duplicate_groups_sampled": len(duplicate_samples),
        "image_context_samples": len(image_samples),
        "empty_stem_units": len(all_empty_stem),
        "pilot_unit_type_counts": dict(sorted(pilot_type_counts.items())),
        "sample_reason_counts": dict(reason_counts.most_common()),
        "population_unit_counts": dict(sorted(input_counts.items())),
        "population_strata": dict(sorted(population_strata.items())),
        "pilot_strata": dict(sorted(pilot_strata.items())),
        "legacy_fields_used": False,
        "seed": seed,
    }
    _write_json_atomic(output / "pilot_report.json", report)
    return report
