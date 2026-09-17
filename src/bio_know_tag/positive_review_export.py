"""Export positive-coverage tasks and DS judgments into one JSON per Label."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                yield value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_band(score: float) -> str:
    if score >= 0.90:
        return ">=0.90"
    if score >= 0.80:
        return "0.80-0.89"
    if score >= 0.70:
        return "0.70-0.79"
    if score >= 0.40:
        return "0.40-0.69"
    if score >= 0.30:
        return "0.30-0.39"
    if score >= 0.20:
        return "0.20-0.29"
    if score >= 0.10:
        return "0.10-0.19"
    if score > 0:
        return "0.01-0.09"
    return "0.00"


def _safe_filename(label_id: str, label_name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", label_name).strip(" ._")
    safe = re.sub(r"\s+", "_", safe)
    if not safe:
        safe = "label"
    return f"{label_id}-{safe[:100]}.json"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["ds_judgment"]["relevance_score"]) for row in rows]
    total = len(rows)
    matches = sum(bool(row["ds_judgment"]["match"]) for row in rows)
    high = sum(score >= 0.80 for score in scores)
    zero = sum(score == 0 for score in scores)
    tiny = sum(0 < score < 0.10 for score in scores)
    irrelevant = zero + tiny
    gray = sum(0.40 <= score < 0.70 for score in scores)
    return {
        "sampled_question_count": total,
        "match_count": matches,
        "match_rate": round(matches / total, 6) if total else None,
        "high_match_count_ge_0_80": high,
        "high_match_rate_ge_0_80": round(high / total, 6) if total else None,
        "basically_irrelevant_count": irrelevant,
        "basically_irrelevant_rate": round(irrelevant / total, 6)
        if total
        else None,
        "basically_irrelevant_count_lt_0_10": irrelevant,
        "basically_irrelevant_rate_lt_0_10": round(irrelevant / total, 6)
        if total
        else None,
        "zero_score_count": zero,
        "score_0_01_0_09_count": tiny,
        "gray_zone_0_40_0_69_count": gray,
        "mean_relevance_score": round(sum(scores) / total, 6) if total else None,
        "minimum_relevance_score": min(scores) if scores else None,
        "maximum_relevance_score": max(scores) if scores else None,
        "score_distribution": dict(
            sorted(Counter(_score_band(score) for score in scores).items())
        ),
    }


def export_positive_review_by_label(
    tasks_path: str | Path,
    results_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Export complete teacher Label cards, questions, and DS judgments."""
    labels = {str(row["label_id"]): row for row in _read_jsonl(labels_path)}
    results: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(results_path):
        task_id = str(row["task_id"])
        if task_id in results:
            raise ValueError(f"duplicate DS result task_id: {task_id}")
        results[task_id] = row
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_tasks: set[str] = set()
    missing_results: list[str] = []
    for task in _read_jsonl(tasks_path):
        task_id = str(task["pair_id"])
        if task_id in seen_tasks:
            raise ValueError(f"duplicate task pair_id: {task_id}")
        seen_tasks.add(task_id)
        result = results.get(task_id)
        if result is None:
            missing_results.append(task_id)
            continue
        label_id = str(task["label_id"])
        if label_id not in labels:
            raise ValueError(f"unknown task label_id: {label_id}")
        if str(result.get("label_id") or label_id) != label_id:
            raise ValueError(f"task/result Label mismatch: {task_id}")
        if str(result.get("question_id") or "") != str(task["question_id"]):
            raise ValueError(f"task/result question mismatch: {task_id}")
        score = float(result["relevance_score"])
        if not 0 <= score <= 1:
            raise ValueError(f"invalid relevance score: {task_id}")
        grouped[label_id].append(
            {
                "pair_id": task_id,
                "question_id": str(task["question_id"]),
                "unit_type": task.get("unit_type"),
                "parent_stem": task.get("parent_stem", ""),
                "stem": task.get("stem", ""),
                "options": task.get("options", ""),
                "answer_text": task.get("answer_text", ""),
                "analysis": task.get("analysis", ""),
                "flags": task.get("flags") or {},
                "expected_relation": task.get("expected_relation"),
                "ds_judgment": {
                    "match": bool(result["match"]),
                    "relevance_score": score,
                    "score_band": _score_band(score),
                    "model": result.get("model"),
                },
            }
        )
    if missing_results:
        raise ValueError(
            f"missing DS results for {len(missing_results)} tasks; first={missing_results[0]}"
        )
    extra_results = set(results) - seen_tasks
    if extra_results:
        raise ValueError(
            f"DS results contain {len(extra_results)} unknown tasks; first={min(extra_results)}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_rows = []
    questions_exported = 0
    for label_id, label in sorted(labels.items()):
        rows = grouped.get(label_id, [])
        rows.sort(
            key=lambda row: (
                float(row["ds_judgment"]["relevance_score"]),
                row["question_id"],
            )
        )
        filename = _safe_filename(label_id, str(label.get("label_name") or ""))
        summary = _summary(rows)
        payload = {
            "schema_version": "positive-label-review-v1",
            "sort_order": "relevance_score_ascending_then_question_id",
            "label": {
                "label_id": label_id,
                "label_name": label.get("label_name", ""),
                "label_path": label.get("label_path", ""),
                "label_type": label.get("label_type", ""),
                "definition": label.get("definition", ""),
                "core_concepts": label.get("core_concepts", ""),
                "common_assessments": label.get("common_assessments", ""),
                "distinctions": label.get("distinctions", ""),
            },
            "summary": summary,
            "questions": rows,
        }
        _write_json(output / filename, payload)
        questions_exported += len(rows)
        index_rows.append(
            {
                "label_id": label_id,
                "label_name": label.get("label_name", ""),
                "file": filename,
                **summary,
            }
        )
    index = {
        "schema_version": "positive-label-review-index-v1",
        "sort_order": "label_id_ascending",
        "input_sha256": {
            "tasks": _sha256(tasks_path),
            "results": _sha256(results_path),
            "labels": _sha256(labels_path),
        },
        "label_count": len(labels),
        "question_count": questions_exported,
        "labels": index_rows,
    }
    _write_json(output / "index.json", index)
    report = {
        "output_dir": str(output),
        "labels_exported": len(labels),
        "labels_with_questions": sum(bool(grouped.get(label_id)) for label_id in labels),
        "labels_without_questions": sum(not grouped.get(label_id) for label_id in labels),
        "questions_exported": questions_exported,
        "index": str(output / "index.json"),
    }
    _write_json(output / "report.json", report)
    return report
