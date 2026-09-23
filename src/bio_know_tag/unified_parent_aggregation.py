"""Aggregate child and parent-material decisions from one mixed prediction stream."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def _rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} must be a JSON object")
            yield row


def aggregate_unified_parent_predictions(
    parent_plans_path: str | Path,
    units_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Compute each real parent's child union plus directly supported extras.

    The unified unit stream determines whether parent material had a model task.
    Missing or unresolved predictions leave a provisional union needing review.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    database_path = output.with_name(f".{output.name}.join.sqlite3.tmp")
    if database_path.exists():
        raise ValueError(f"temporary join database already exists: {database_path}")
    connection = sqlite3.connect(database_path)
    counts: Counter[str] = Counter()
    try:
        connection.execute("CREATE TABLE units (question_id TEXT PRIMARY KEY, parent_id TEXT, unit_type TEXT)")
        connection.execute("CREATE TABLE predictions (question_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for unit in _rows(units_path):
            question_id = str(unit.get("question_id") or "")
            if not question_id:
                raise ValueError("unit missing question_id")
            connection.execute(
                "INSERT INTO units VALUES (?, ?, ?)",
                (question_id, str(unit.get("parent_id") or ""), str(unit.get("unit_type") or "")),
            )
        for prediction in _rows(predictions_path):
            question_id = str(prediction.get("question_id") or "")
            if not question_id:
                raise ValueError("prediction missing question_id")
            indexed = connection.execute(
                "SELECT parent_id, unit_type FROM units WHERE question_id = ?", (question_id,)
            ).fetchone()
            if indexed is None:
                raise ValueError(f"prediction has no unit: {question_id}")
            if str(prediction.get("unit_type") or "") != indexed[1]:
                raise ValueError(f"prediction unit_type mismatch: {question_id}")
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?)",
                (question_id, json.dumps(prediction, ensure_ascii=False)),
            )
        connection.commit()

        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for plan in _rows(parent_plans_path):
                parent_id = str(plan.get("question_id") or "")
                if not parent_id:
                    raise ValueError("parent plan missing question_id")
                counts["parents_scanned"] += 1
                child_ids = list(dict.fromkeys(str(value) for value in plan.get("child_question_ids") or []))
                child_union: dict[str, dict[str, Any]] = {}
                missing_children: list[str] = []
                child_needs_review = False
                child_training_eligible = True
                for child_id in child_ids:
                    indexed = connection.execute(
                        "SELECT parent_id, unit_type FROM units WHERE question_id = ?", (child_id,)
                    ).fetchone()
                    if indexed is None:
                        missing_children.append(child_id)
                        continue
                    if indexed[0] != parent_id or indexed[1] not in {"sub_question", "orphan_sub_question"}:
                        raise ValueError(f"parent plan child unit mismatch: {parent_id}::{child_id}")
                    fetched = connection.execute(
                        "SELECT payload FROM predictions WHERE question_id = ?", (child_id,)
                    ).fetchone()
                    if fetched is None:
                        missing_children.append(child_id)
                        continue
                    child = json.loads(fetched[0])
                    child_needs_review |= bool(child.get("needs_review"))
                    child_training_eligible &= bool(child.get("usable_for_training"))
                    for label in child.get("selected_labels") or []:
                        label_id = str(label["label_id"])
                        if label_id not in child_union:
                            child_union[label_id] = {
                                "label_id": label_id,
                                "label_name": str(label.get("label_name") or ""),
                                "source_question_ids": [],
                            }
                        child_union[label_id]["source_question_ids"].append(child_id)

                indexed_parent = connection.execute(
                    "SELECT unit_type FROM units WHERE question_id = ?", (parent_id,)
                ).fetchone()
                if indexed_parent is not None and indexed_parent[0] != "composite_parent_extra":
                    raise ValueError(f"parent material unit_type mismatch: {parent_id}")
                has_parent_unit = indexed_parent is not None
                fetched_parent = connection.execute(
                    "SELECT payload FROM predictions WHERE question_id = ?", (parent_id,)
                ).fetchone()
                parent_prediction_missing = has_parent_unit and fetched_parent is None
                parent_prediction = json.loads(fetched_parent[0]) if fetched_parent else None
                parent_needs_review = bool(parent_prediction and parent_prediction.get("needs_review"))
                parent_training_eligible = not has_parent_unit or bool(
                    parent_prediction and not parent_prediction_missing and not parent_needs_review
                )
                extras: list[dict[str, Any]] = []
                extra_ids: set[str] = set()
                for label in (parent_prediction or {}).get("selected_labels") or []:
                    label_id = str(label["label_id"])
                    if label_id not in child_union and label_id not in extra_ids:
                        extras.append(label)
                        extra_ids.add(label_id)
                child_labels = list(child_union.values())
                knowledge_labels = child_labels + extras
                incomplete = bool(missing_children or parent_prediction_missing)
                needs_review = incomplete or child_needs_review or parent_needs_review
                result = {
                    "question_id": parent_id,
                    "unit_type": "composite_parent",
                    "child_question_ids": child_ids,
                    "child_union_labels": child_labels,
                    "parent_extra_labels": extras,
                    "knowledge_labels": knowledge_labels,
                    "missing_child_question_ids": missing_children,
                    "parent_extra_prediction_missing": parent_prediction_missing,
                    "needs_review": needs_review,
                    "usable_for_training": bool(
                        knowledge_labels and not needs_review
                        and child_training_eligible and parent_training_eligible
                    ),
                }
                destination.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                counts["incomplete_parents" if incomplete else "complete_parents"] += 1
                counts["parents_needing_review"] += int(needs_review)
                counts["parent_extra_labels"] += len(extras)
        temporary.replace(output)
        return {
            "parents_scanned": counts["parents_scanned"],
            "complete_parents": counts["complete_parents"],
            "incomplete_parents": counts["incomplete_parents"],
            "parents_needing_review": counts["parents_needing_review"],
            "parent_extra_labels": counts["parent_extra_labels"],
        }
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()
        if temporary.exists():
            temporary.unlink()
