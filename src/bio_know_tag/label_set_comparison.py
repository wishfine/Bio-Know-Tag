"""Compare selected Label sets with current-catalog historical knw_ids."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = (
    "exact",
    "contains_legacy",
    "subset_of_legacy",
    "mixed_add_remove",
    "disjoint",
)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_ids(row: dict[str, Any]) -> set[str]:
    value = row.get("selected_labels")
    if value is None and isinstance(row.get("prediction"), dict):
        value = row["prediction"].get("selected_labels")
    if not isinstance(value, list):
        raise ValueError(f"prediction {row.get('question_id')} has no selected_labels list")
    result = set()
    for label in value:
        label_id = label.get("label_id") if isinstance(label, dict) else label
        if not label_id:
            raise ValueError(f"prediction {row.get('question_id')} has a blank Label ID")
        result.add(str(label_id))
    return result


def relationship(legacy: set[str], predicted: set[str]) -> str:
    """Five mutually exclusive categories for a nonempty legacy set."""
    if not legacy:
        raise ValueError("legacy set must be nonempty")
    if predicted == legacy:
        return "exact"
    if legacy < predicted:
        return "contains_legacy"
    if predicted < legacy:
        return "subset_of_legacy"
    if legacy.isdisjoint(predicted):
        return "disjoint"
    return "mixed_add_remove"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percent(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def build_legacy_id_snapshot(
    *, sample_units_path: str | Path, full_units_path: str | Path,
    output_path: str | Path, expected_units: int | None = None,
) -> dict[str, Any]:
    """Recover old IDs for a blind pilot from the full label-unit file once."""
    question_ids: set[str] = set()
    for row in read_jsonl(sample_units_path):
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in question_ids:
            raise ValueError(f"missing/duplicate sample question ID: {question_id!r}")
        question_ids.add(question_id)
    if expected_units is not None and len(question_ids) != expected_units:
        raise ValueError(f"expected {expected_units} sample units, found {len(question_ids)}")
    recovered: dict[str, list[str]] = {}
    scanned = 0
    for row in read_jsonl(full_units_path):
        scanned += 1
        question_id = str(row.get("question_id") or "")
        if question_id not in question_ids:
            continue
        if question_id in recovered:
            raise ValueError(f"duplicate full-unit question ID: {question_id}")
        values = row.get("legacy_knw_ids", row.get("knw_ids"))
        if not isinstance(values, list):
            raise ValueError(f"full unit {question_id} has no legacy ID list")
        recovered[question_id] = [str(value) for value in values if value]
    missing = question_ids - set(recovered)
    if missing:
        raise ValueError(f"{len(missing)} sample IDs not found in full units: {sorted(missing)[:5]}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for question_id in sorted(recovered):
            handle.write(json.dumps({
                "question_id": question_id, "legacy_knw_ids": recovered[question_id]
            }, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    report = {
        "sample_units": len(question_ids), "full_units_scanned": scanned,
        "matched_units": len(recovered), "output": str(destination),
        "sample_sha256": sha256(sample_units_path),
        "full_units_sha256": sha256(full_units_path),
    }
    _atomic_json(destination.with_suffix(".report.json"), report)
    return report


def compare_label_sets(
    *,
    units_path: str | Path,
    predictions_path: str | Path,
    labels_path: str | Path,
    legacy_snapshot_path: str | Path | None = None,
    output_dir: str | Path,
    expected_units: int | None = None,
) -> dict[str, Any]:
    """Write question-level and per-Label comparisons for one model/run."""
    labels = {}
    for row in read_jsonl(labels_path):
        label_id = str(row.get("label_id") or "")
        if not label_id or label_id in labels:
            raise ValueError(f"invalid/duplicate catalog Label ID: {label_id!r}")
        labels[label_id] = row
    current_ids = set(labels)

    legacy_snapshot = None
    if legacy_snapshot_path is not None:
        legacy_snapshot = {}
        for row in read_jsonl(legacy_snapshot_path):
            question_id = str(row.get("question_id") or "")
            values = row.get("legacy_knw_ids")
            if not question_id or question_id in legacy_snapshot or not isinstance(values, list):
                raise ValueError(f"invalid/duplicate legacy snapshot row: {question_id!r}")
            legacy_snapshot[question_id] = values

    predictions: dict[str, tuple[set[str], dict[str, Any]]] = {}
    for row in read_jsonl(predictions_path):
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in predictions:
            raise ValueError(f"missing/duplicate prediction question ID: {question_id!r}")
        selected = selected_ids(row)
        unknown = selected - current_ids
        if unknown:
            raise ValueError(f"prediction {question_id} has out-of-catalog Label IDs: {sorted(unknown)}")
        predictions[question_id] = (selected, row)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    overall = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    seen_units: set[str] = set()
    used_predictions: set[str] = set()
    question_path = output / "per_question.jsonl"
    with question_path.open("w", encoding="utf-8") as question_output:
        for unit in read_jsonl(units_path):
            question_id = str(unit.get("question_id") or "")
            if not question_id or question_id in seen_units:
                raise ValueError(f"missing/duplicate unit question ID: {question_id!r}")
            seen_units.add(question_id)
            overall["units"] += 1
            unit_type = str(unit.get("unit_type") or "unknown")
            if legacy_snapshot is not None:
                if question_id not in legacy_snapshot:
                    raise ValueError(f"unit {question_id} absent from legacy snapshot")
                raw_legacy = legacy_snapshot[question_id]
            else:
                raw_legacy = unit.get("legacy_knw_ids", unit.get("knw_ids"))
            if not isinstance(raw_legacy, list):
                raise ValueError(f"unit {question_id} has no legacy ID list; pass --legacy-snapshot")
            legacy_all = {str(value) for value in raw_legacy if value}
            legacy = legacy_all & current_ids
            obsolete = legacy_all - current_ids
            overall["obsolete_assignments_ignored"] += len(obsolete)
            overall["duplicate_legacy_assignments_removed"] += len(raw_legacy) - len(legacy_all)
            overall["duplicate_content_units"] += bool((unit.get("flags") or {}).get("duplicate_of"))
            if question_id not in predictions:
                overall["missing_prediction"] += 1
                by_type[unit_type]["missing_prediction"] += 1
                continue
            used_predictions.add(question_id)
            predicted, prediction = predictions[question_id]
            overall["compared"] += 1
            by_type[unit_type]["compared"] += 1
            if not legacy:
                category = "no_current_legacy"
                overall[category] += 1
                by_type[unit_type][category] += 1
                overall["no_current_legacy_predicted_nonempty"] += bool(predicted)
            else:
                category = relationship(legacy, predicted)
                overall["eligible"] += 1
                overall[category] += 1
                by_type[unit_type]["eligible"] += 1
                by_type[unit_type][category] += 1
                overall["legacy_assignments"] += len(legacy)
                overall["retained_assignments"] += len(legacy & predicted)
                overall["missing_assignments"] += len(legacy - predicted)
                overall["added_assignments"] += len(predicted - legacy)
                for label_id in legacy:
                    stats = per_label[label_id]
                    stats["legacy_questions"] += 1
                    stats[category] += 1
                    stats["retained"] += label_id in predicted
                    stats["missed"] += label_id not in predicted
            for label_id in predicted:
                stats = per_label[label_id]
                stats["predicted_questions"] += 1
                stats["new_vs_legacy"] += label_id not in legacy
            question_output.write(json.dumps({
                "question_id": question_id,
                "unit_type": unit_type,
                "category": category,
                "legacy_label_ids_current": sorted(legacy),
                "obsolete_legacy_id_count": len(obsolete),
                "predicted_label_ids": sorted(predicted),
                "retained_label_ids": sorted(legacy & predicted),
                "removed_label_ids": sorted(legacy - predicted),
                "added_label_ids": sorted(predicted - legacy),
                "needs_review": bool(prediction.get("needs_review")),
                "usable_for_training": bool(prediction.get("usable_for_training")),
            }, ensure_ascii=False) + "\n")

    if expected_units is not None and overall["units"] != expected_units:
        raise ValueError(f"expected {expected_units} units, found {overall['units']}")
    if legacy_snapshot is not None and set(legacy_snapshot) != seen_units:
        raise ValueError("legacy snapshot question IDs differ from sample units")
    extra_prediction_ids = set(predictions) - used_predictions
    if extra_prediction_ids:
        raise ValueError(f"{len(extra_prediction_ids)} predictions absent from units; examples: {sorted(extra_prediction_ids)[:5]}")

    label_path = output / "per_label.jsonl"
    with label_path.open("w", encoding="utf-8") as handle:
        for label_id, label in sorted(labels.items()):
            stats = per_label[label_id]
            old = stats["legacy_questions"]
            predicted = stats["predicted_questions"]
            row = {
                "label_id": label_id,
                "label_name": label.get("label_name", ""),
                "legacy_questions": old,
                "predicted_questions": predicted,
                "retained": stats["retained"],
                "missed": stats["missed"],
                "new_vs_legacy": stats["new_vs_legacy"],
                "retention_rate": _percent(stats["retained"], old),
                "miss_rate": _percent(stats["missed"], old),
                "predicted_supported_by_legacy_rate": _percent(stats["retained"], predicted),
                "relationship_counts_among_legacy_questions": {
                    category: stats[category] for category in CATEGORIES
                },
                "relationship_rates_among_legacy_questions": {
                    category: _percent(stats[category], old) for category in CATEGORIES
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "catalog_labels": len(labels),
        "predictions_loaded": len(predictions),
        "counts": dict(overall),
        "category_rates_among_eligible": {
            category: _percent(overall[category], overall["eligible"]) for category in CATEGORIES
        },
        "by_unit_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "input_sha256": {
            "units": sha256(units_path),
            "predictions": sha256(predictions_path),
            "labels": sha256(labels_path),
            "legacy_snapshot": sha256(legacy_snapshot_path) if legacy_snapshot_path is not None else None,
        },
        "warning": "Historical knw_ids are weak supervision, not verified gold labels. Agreement is not accuracy.",
    }
    _atomic_json(output / "report.json", report)
    return report
