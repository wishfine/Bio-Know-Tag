"""Use current-taxonomy legacy IDs as weak supervision for retrieval experiments."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.retrieval import format_label_path


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    labels = {}
    for row in _read_jsonl(path):
        label_id = str(row.get("label_id") or "").strip()
        if label_id:
            labels[label_id] = row
    if not labels:
        raise ValueError("current label catalog is empty")
    return labels


def valid_legacy_targets(
    unit: dict[str, Any], current_label_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Split historical IDs into current-458 targets and retired taxonomy IDs."""
    legacy_ids = _unique(
        unit.get("legacy_knw_ids", unit.get("knw_ids", [])) or []
    )
    valid = [label_id for label_id in legacy_ids if label_id in current_label_ids]
    obsolete = [label_id for label_id in legacy_ids if label_id not in current_label_ids]
    return valid, obsolete


def build_legacy_recall_sample(
    units_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    sample_rate: float = 1.0,
    seed: str = "legacy-recall-v1",
) -> dict[str, Any]:
    """Deterministically hash-sample units while retaining legacy target metadata."""
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    labels = _load_labels(labels_path)
    label_ids = set(labels)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_units.jsonl"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    scanned = selected = eligible = obsolete_only = no_legacy = 0
    valid_assignments = obsolete_assignments = 0
    threshold = int(sample_rate * (2**64))
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for unit in _read_jsonl(units_path):
            scanned += 1
            question_id = str(unit.get("question_id") or "")
            score = int.from_bytes(
                hashlib.sha256(f"{seed}:{question_id}".encode()).digest()[:8],
                "big",
            )
            if score >= threshold:
                continue
            selected += 1
            valid, obsolete = valid_legacy_targets(unit, label_ids)
            valid_assignments += len(valid)
            obsolete_assignments += len(obsolete)
            if valid:
                eligible += 1
            elif obsolete:
                obsolete_only += 1
            else:
                no_legacy += 1
            if valid:
                row = dict(unit)
                row["legacy_eval_label_ids"] = valid
                row["obsolete_legacy_id_count"] = len(obsolete)
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    report = {
        "units_scanned": scanned,
        "sampled_units": selected,
        "sample_rate": sample_rate,
        "seed": seed,
        "eligible_questions": eligible,
        "obsolete_only_questions": obsolete_only,
        "no_legacy_questions": no_legacy,
        "valid_legacy_assignments": valid_assignments,
        "obsolete_legacy_assignments_ignored": obsolete_assignments,
        "current_label_count": len(labels),
        "note": "Only legacy IDs present in the current 458-label catalog are weak-supervision targets.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def evaluate_legacy_recall(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    ks: tuple[int, ...] = (5, 10, 20, 25),
) -> dict[str, Any]:
    """Measure candidate coverage against valid historical IDs as weak labels."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    ks = tuple(sorted(set(ks)))
    labels = _load_labels(labels_path)
    label_ids = set(labels)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    misses_path = output_dir / "misses.jsonl"
    temporary = misses_path.with_name(f".{misses_path.name}.tmp")
    scanned = evaluated = obsolete_only = no_legacy = missing_candidates = 0
    obsolete_assignments = 0
    totals = {k: Counter() for k in ks}
    by_unit_type: dict[str, dict[int, Counter[str]]] = defaultdict(
        lambda: {k: Counter() for k in ks}
    )
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    with temporary.open("w", encoding="utf-8", newline="\n") as misses:
        pairs = itertools.zip_longest(
            _read_jsonl(units_path), _read_jsonl(candidates_path)
        )
        for unit, candidate_row in pairs:
            if unit is None or candidate_row is None:
                raise ValueError("units and candidates must contain the same number of rows")
            scanned += 1
            question_id = str(unit.get("question_id") or "")
            if question_id != str(candidate_row.get("question_id") or ""):
                raise ValueError(f"units/candidates order mismatch at {question_id}")
            valid = _unique(unit.get("legacy_eval_label_ids") or [])
            if not valid:
                valid, obsolete = valid_legacy_targets(unit, label_ids)
            else:
                _, obsolete = valid_legacy_targets(unit, label_ids)
            obsolete_assignments += len(obsolete)
            if not valid:
                if obsolete:
                    obsolete_only += 1
                else:
                    no_legacy += 1
                continue
            evaluated += 1
            unit_type = str(unit.get("unit_type") or "unknown")
            ranked = _unique(
                item.get("label_id")
                for item in (candidate_row.get("candidates") or [])
            )
            for label_id in valid:
                per_label[label_id]["targets"] += 1
            for k in ks:
                hits = set(valid) & set(ranked[:k])
                totals[k]["any_hit"] += int(bool(hits))
                totals[k]["all_hit"] += int(len(hits) == len(valid))
                totals[k]["target_assignments"] += len(valid)
                totals[k]["hit_assignments"] += len(hits)
                by_unit_type[unit_type][k]["questions"] += 1
                by_unit_type[unit_type][k]["any_hit"] += int(bool(hits))
                by_unit_type[unit_type][k]["all_hit"] += int(len(hits) == len(valid))
                by_unit_type[unit_type][k]["target_assignments"] += len(valid)
                by_unit_type[unit_type][k]["hit_assignments"] += len(hits)
                for label_id in hits:
                    per_label[label_id][f"hit_at_{k}"] += 1
            largest_hits = set(valid) & set(ranked[: ks[-1]])
            if len(largest_hits) != len(valid):
                missing = [label_id for label_id in valid if label_id not in largest_hits]
                misses.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "valid_legacy_ids": valid,
                            "missing_valid_legacy_ids": missing,
                            "candidate_label_ids": ranked[: ks[-1]],
                            "unit_type": unit.get("unit_type"),
                            "stem": unit.get("stem", ""),
                            "answer_text": unit.get("answer_text", ""),
                            "analysis": unit.get("analysis", ""),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    temporary.replace(misses_path)
    metrics = {}
    for k in ks:
        counter = totals[k]
        denominator = evaluated or 1
        assignments = counter["target_assignments"] or 1
        metrics[str(k)] = {
            "any_hit": counter["any_hit"],
            "any_hit_rate": round(counter["any_hit"] / denominator, 6),
            "all_hit": counter["all_hit"],
            "all_hit_rate": round(counter["all_hit"] / denominator, 6),
            "hit_assignments": counter["hit_assignments"],
            "target_assignments": counter["target_assignments"],
            "micro_recall": round(counter["hit_assignments"] / assignments, 6),
        }
    per_label_rows = []
    for label_id, counter in per_label.items():
        targets = counter["targets"]
        row = {
            "label_id": label_id,
            "label_name": labels[label_id].get("label_name", ""),
            "targets": targets,
        }
        for k in ks:
            hits = counter[f"hit_at_{k}"]
            row[f"hits_at_{k}"] = hits
            row[f"recall_at_{k}"] = round(hits / targets, 6)
        per_label_rows.append(row)
    per_label_rows.sort(key=lambda row: (row[f"recall_at_{ks[-1]}"], -row["targets"], row["label_id"]))
    per_label_path = output_dir / "per_label.jsonl"
    with per_label_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in per_label_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    unit_type_metrics = {}
    for unit_type, counters_by_k in sorted(by_unit_type.items()):
        unit_type_metrics[unit_type] = {}
        for k, counter in counters_by_k.items():
            questions = counter["questions"] or 1
            assignments = counter["target_assignments"] or 1
            unit_type_metrics[unit_type][str(k)] = {
                "questions": counter["questions"],
                "any_hit_rate": round(counter["any_hit"] / questions, 6),
                "all_hit_rate": round(counter["all_hit"] / questions, 6),
                "micro_recall": round(counter["hit_assignments"] / assignments, 6),
            }
    report = {
        "questions_scanned": scanned,
        "questions_evaluated": evaluated,
        "obsolete_only_questions": obsolete_only,
        "no_legacy_questions": no_legacy,
        "obsolete_legacy_assignments_ignored": obsolete_assignments,
        "missing_candidate_rows": missing_candidates,
        "ks": list(ks),
        "metrics": metrics,
        "metrics_by_unit_type": unit_type_metrics,
        "labels_evaluated": len(per_label_rows),
        "ground_truth_warning": "Historical IDs are weak supervision, not verified gold labels.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def augment_candidates_with_legacy(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    legacy_units_path: str | Path | None = None,
    max_legacy_additions: int | None = None,
) -> dict[str, Any]:
    """Append only current-taxonomy legacy IDs to an existing candidate sidecar."""
    if max_legacy_additions is not None and max_legacy_additions < 1:
        raise ValueError("max_legacy_additions must be positive")
    labels = _load_labels(labels_path)
    label_ids = set(labels)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidates.jsonl"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    questions = additions = already_present = obsolete_removed = truncated = 0
    legacy_source_rows_scanned = 0
    legacy_by_id: dict[str, dict[str, Any]] = {}
    target_question_ids: set[str] = set()
    if legacy_units_path is not None:
        target_question_ids = {
            str(unit.get("question_id") or "") for unit in _read_jsonl(units_path)
        }
        target_question_ids.discard("")
        for source_unit in _read_jsonl(legacy_units_path):
            legacy_source_rows_scanned += 1
            question_id = str(source_unit.get("question_id") or "")
            if question_id in target_question_ids:
                legacy_by_id[question_id] = source_unit
    count_distribution: Counter[str] = Counter()
    units = _read_jsonl(units_path)
    candidates = _read_jsonl(candidates_path)
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for unit, row in itertools.zip_longest(units, candidates):
            if unit is None or row is None:
                raise ValueError("units and candidates must contain the same number of rows")
            question_id = str(unit.get("question_id") or "")
            if question_id != str(row.get("question_id") or ""):
                raise ValueError(f"units/candidates order mismatch at {question_id}")
            questions += 1
            legacy_unit = legacy_by_id.get(question_id, unit)
            valid, obsolete = valid_legacy_targets(legacy_unit, label_ids)
            obsolete_removed += len(obsolete)
            items = [dict(item) for item in (row.get("candidates") or [])]
            by_id = {str(item.get("label_id")): item for item in items}
            missing = [label_id for label_id in valid if label_id not in by_id]
            if max_legacy_additions is not None and len(missing) > max_legacy_additions:
                truncated += len(missing) - max_legacy_additions
                missing = missing[:max_legacy_additions]
            for label_id in valid:
                existing = by_id.get(label_id)
                if existing is not None:
                    sources = list(existing.get("sources") or [])
                    if "legacy" not in sources:
                        sources.append("legacy")
                    existing["sources"] = sources
                    already_present += 1
            for label_id in missing:
                label = labels[label_id]
                item = {
                    "label_id": label_id,
                    "label_name": str(label.get("label_name") or ""),
                    "label_path": format_label_path(label.get("label_path")),
                    "candidate_rank": len(items) + 1,
                    "sources": ["legacy"],
                    "sparse_rank": None,
                    "sparse_score": None,
                    "dense_rank": None,
                    "dense_score": None,
                }
                items.append(item)
                by_id[label_id] = item
                additions += 1
            for rank, item in enumerate(items, 1):
                item["candidate_rank"] = rank
            count_distribution[str(len(items))] += 1
            augmented = dict(row)
            augmented["method"] = f"{row.get('method', 'candidates')}+valid_legacy"
            augmented["retrieval_version"] = f"{row.get('retrieval_version', 'unknown')}+legacy-current458-v1"
            augmented["candidates"] = items
            output.write(json.dumps(augmented, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    report = {
        "input": questions,
        "processed": questions,
        "error": 0,
        "legacy_candidates_added": additions,
        "valid_legacy_candidates_already_present": already_present,
        "obsolete_legacy_assignments_removed": obsolete_removed,
        "legacy_additions_truncated": truncated,
        "max_legacy_additions": max_legacy_additions,
        "legacy_units_path": str(legacy_units_path) if legacy_units_path else None,
        "legacy_source_rows_scanned": legacy_source_rows_scanned,
        "legacy_source_questions_matched": len(legacy_by_id),
        "legacy_source_questions_missing": (
            len(target_question_ids - set(legacy_by_id))
            if legacy_units_path is not None
            else 0
        ),
        "candidate_count_distribution": dict(sorted(count_distribution.items(), key=lambda item: int(item[0]))),
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
