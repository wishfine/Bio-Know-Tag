"""Build and analyze verified sibling-Label hard-negative experiments."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MATCH_THRESHOLD = 0.70


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                yield value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _label_parent(label: dict[str, Any]) -> str:
    path = str(label.get("label_path") or "").replace("->", "@")
    return path.rsplit("@", 1)[0] if "@" in path else ""


def _priority(seed: str, target: str, source: str, question_id: str) -> int:
    return int(
        hashlib.sha256(
            f"{seed}:{target}:{source}:{question_id}".encode()
        ).hexdigest(),
        16,
    )


def _keep_candidate(
    heap: list[tuple[int, str, dict[str, Any], float]],
    task: dict[str, Any],
    score: float,
    *,
    target_label_id: str,
    source_label_id: str,
    seed: str,
    limit: int,
) -> None:
    question_id = str(task["question_id"])
    priority = _priority(seed, target_label_id, source_label_id, question_id)
    entry = (-priority, question_id, task, score)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif priority < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def build_verified_sibling_hard_negatives(
    positive_tasks_path: str | Path,
    positive_results_path: str | Path,
    units_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    negatives_per_label: int = 50,
    min_source_score: float = 0.80,
    seed: str = "verified-sibling-hard-negative-v1",
    require_complete: bool = True,
) -> dict[str, Any]:
    """Sample target-Label negatives from high-confidence sibling positives."""
    if negatives_per_label < 1:
        raise ValueError("negatives_per_label must be positive")
    if not 0 <= min_source_score <= 1:
        raise ValueError("min_source_score must be between 0 and 1")
    labels = {
        str(row["label_id"]): row
        for row in _read_jsonl(labels_path)
        if str(row.get("label_id") or "")
    }
    parent_groups: dict[str, list[str]] = defaultdict(list)
    for label_id, label in labels.items():
        parent_groups[_label_parent(label)].append(label_id)
    siblings: dict[str, list[str]] = {}
    for label_id, label in labels.items():
        siblings[label_id] = sorted(
            other
            for other in parent_groups[_label_parent(label)]
            if other != label_id
        )

    tasks = {
        str(task["pair_id"]): task for task in _read_jsonl(positive_tasks_path)
    }
    results = {
        str(row["task_id"]): row for row in _read_jsonl(positive_results_path)
    }
    completed = len(set(tasks).intersection(results))
    if require_complete and completed != len(tasks):
        raise ValueError(
            f"positive results are incomplete: {completed}/{len(tasks)}"
        )
    verified_sources: list[tuple[dict[str, Any], float]] = []
    for task_id, task in tasks.items():
        result = results.get(task_id)
        if result is None:
            continue
        score = float(result.get("relevance_score", 0))
        if (
            str(task.get("unit_type") or "") == "standalone"
            and score >= min_source_score
        ):
            verified_sources.append((task, score))

    needed_question_ids = {
        str(task["question_id"]) for task, _ in verified_sources
    }
    legacy_by_question: dict[str, set[str]] = {}
    units_scanned = 0
    for unit in _read_jsonl(units_path):
        units_scanned += 1
        question_id = str(unit.get("question_id") or "")
        if question_id not in needed_question_ids:
            continue
        legacy_by_question[question_id] = {
            str(label_id)
            for label_id in (
                unit.get("legacy_candidate_ids")
                or unit.get("legacy_knw_ids")
                or []
            )
            if str(label_id) in labels
        }

    pools: dict[
        tuple[str, str], list[tuple[int, str, dict[str, Any], float]]
    ] = defaultdict(list)
    target_already_present = missing_unit_rows = no_sibling_sources = 0
    for task, score in verified_sources:
        source_label_id = str(task["label_id"])
        target_ids = siblings.get(source_label_id, [])
        if not target_ids:
            no_sibling_sources += 1
            continue
        question_id = str(task["question_id"])
        legacy_ids = legacy_by_question.get(question_id)
        if legacy_ids is None:
            missing_unit_rows += 1
            continue
        for target_label_id in target_ids:
            if target_label_id in legacy_ids:
                target_already_present += 1
                continue
            _keep_candidate(
                pools[(target_label_id, source_label_id)],
                task,
                score,
                target_label_id=target_label_id,
                source_label_id=source_label_id,
                seed=seed,
                limit=negatives_per_label,
            )

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hard_negative_samples.jsonl"
    selected_rows: list[dict[str, Any]] = []
    target_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    for target_label_id in labels:
        source_ids = sorted(
            source
            for target, source in pools
            if target == target_label_id
        )
        queues = {
            source: sorted(
                pools[(target_label_id, source)],
                key=lambda entry: (-entry[0], entry[1]),
            )
            for source in source_ids
        }
        positions = Counter()
        selected_by_question: dict[str, dict[str, Any]] = {}
        while len(selected_by_question) < negatives_per_label:
            progressed = False
            for source_label_id in source_ids:
                position = positions[source_label_id]
                queue = queues[source_label_id]
                if position >= len(queue):
                    continue
                progressed = True
                positions[source_label_id] += 1
                _, question_id, task, score = queue[position]
                existing = selected_by_question.get(question_id)
                if existing is not None:
                    if source_label_id not in existing["source_label_ids"]:
                        existing["source_label_ids"].append(source_label_id)
                        existing["source_label_names"].append(
                            labels[source_label_id].get("label_name", "")
                        )
                    existing["source_positive_score"] = max(
                        existing["source_positive_score"], score
                    )
                    continue
                selected_by_question[question_id] = {
                    "pair_id": f"{question_id}::{target_label_id}",
                    "question_id": question_id,
                    "label_id": target_label_id,
                    "expected_relation": "verified_sibling_hard_negative",
                    "unit_type": task.get("unit_type"),
                    "parent_stem": task.get("parent_stem", ""),
                    "stem": task.get("stem", ""),
                    "options": task.get("options", ""),
                    "answer_text": task.get("answer_text", ""),
                    "analysis": task.get("analysis", ""),
                    "source_label_ids": [source_label_id],
                    "source_label_names": [
                        labels[source_label_id].get("label_name", "")
                    ],
                    "source_positive_score": score,
                }
                if len(selected_by_question) >= negatives_per_label:
                    break
            if not progressed:
                break
        for row in selected_by_question.values():
            row["source_label_ids"].sort()
            row["source_label_names"] = [
                labels[source_id].get("label_name", "")
                for source_id in row["source_label_ids"]
            ]
            selected_rows.append(row)
            target_distribution[target_label_id] += 1
            source_distribution.update(row["source_label_ids"])
    selected_rows.sort(key=lambda row: (row["label_id"], row["question_id"]))
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "positive_tasks": len(tasks),
        "positive_results_completed": completed,
        "positive_results_missing": len(tasks) - completed,
        "min_source_score": min_source_score,
        "verified_source_pairs": len(verified_sources),
        "verified_source_questions": len(needed_question_ids),
        "units_scanned": units_scanned,
        "verified_questions_found_in_units": len(legacy_by_question),
        "missing_verified_question_units": missing_unit_rows,
        "source_pairs_without_siblings": no_sibling_sources,
        "target_already_in_legacy_excluded": target_already_present,
        "negatives_per_label": negatives_per_label,
        "samples": len(selected_rows),
        "labels_with_hard_negatives": len(target_distribution),
        "labels_without_hard_negatives": len(labels) - len(target_distribution),
        "candidate_count_distribution": dict(
            sorted(Counter(target_distribution.values()).items())
        ),
        "seed": seed,
        "warning": "A missing historical target Label is not gold-negative; source questions are verified positives of sibling Labels and remain hard-negative candidates pending DS/human review.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def _false_accept_tier(rate: float) -> str:
    if rate <= 0.05:
        return "STABLE_<=5%"
    if rate <= 0.15:
        return "MINOR_5_15%"
    if rate <= 0.30:
        return "BROAD_OR_OVERLAP_15_30%"
    return "SEVERE_>30%"


def analyze_hard_negative_results(
    samples_path: str | Path,
    results_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Measure target-Label false acceptance on verified sibling negatives."""
    labels = {
        str(row["label_id"]): row for row in _read_jsonl(labels_path)
    }
    samples = {
        str(row["pair_id"]): row for row in _read_jsonl(samples_path)
    }
    results = {
        str(row["task_id"]): row for row in _read_jsonl(results_path)
    }
    completed = {
        task_id: result
        for task_id, result in results.items()
        if task_id in samples
    }
    by_target: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_pair: dict[
        tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    false_accept_samples: list[dict[str, Any]] = []
    for task_id, result in completed.items():
        sample = samples[task_id]
        target = str(sample["label_id"])
        by_target[target].append((sample, result))
        for source in sample.get("source_label_ids") or []:
            by_pair[(str(source), target)].append((sample, result))
        if float(result["relevance_score"]) >= MATCH_THRESHOLD:
            false_accept_samples.append(
                {
                    **sample,
                    "relevance_score": float(result["relevance_score"]),
                    "match": True,
                }
            )
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_label_rows = []
    for target, values in sorted(by_target.items()):
        accepted = sum(
            float(result["relevance_score"]) >= MATCH_THRESHOLD
            for _, result in values
        )
        rate = accepted / len(values)
        per_label_rows.append(
            {
                "label_id": target,
                "label_name": labels.get(target, {}).get("label_name", ""),
                "hard_negative_total": len(values),
                "false_accept": accepted,
                "false_accept_rate": round(rate, 6),
                "boundary_tier": _false_accept_tier(rate),
            }
        )
    with (output_dir / "per_label.jsonl").open("w", encoding="utf-8") as output:
        for row in per_label_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    confusion_rows = []
    for (source, target), values in sorted(by_pair.items()):
        accepted = sum(
            float(result["relevance_score"]) >= MATCH_THRESHOLD
            for _, result in values
        )
        confusion_rows.append(
            {
                "source_label_id": source,
                "source_label_name": labels.get(source, {}).get("label_name", ""),
                "target_label_id": target,
                "target_label_name": labels.get(target, {}).get("label_name", ""),
                "total": len(values),
                "false_accept": accepted,
                "false_accept_rate": round(accepted / len(values), 6),
            }
        )
    with (output_dir / "confusion_pairs.jsonl").open("w", encoding="utf-8") as output:
        for row in confusion_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "false_accept_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in sorted(
            false_accept_samples,
            key=lambda item: (item["label_id"], -item["relevance_score"], item["question_id"]),
        ):
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    accepted_total = len(false_accept_samples)
    report = {
        "planned_tasks": len(samples),
        "completed_tasks": len(completed),
        "pending_tasks": len(samples) - len(completed),
        "labels_planned": len({str(row["label_id"]) for row in samples.values()}),
        "labels_completed": len(by_target),
        "false_accept": accepted_total,
        "false_accept_rate": round(accepted_total / len(completed), 6)
        if completed
        else None,
        "boundary_tier_counts": dict(
            sorted(Counter(row["boundary_tier"] for row in per_label_rows).items())
        ),
        "warning": "False acceptance on verified sibling positives diagnoses boundary overlap; historical non-assignment alone is not treated as gold-negative.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def _combined_screen(
    positive_count: int,
    positive_rate: float | None,
    negative_count: int,
    false_accept_rate: float | None,
) -> str:
    if positive_count < 300:
        return "U_LONG_TAIL_REVIEW"
    if negative_count < 20 or false_accept_rate is None:
        return "U_INSUFFICIENT_HARD_NEGATIVES"
    if positive_rate is None:
        return "U_INCOMPLETE_POSITIVE_RESULTS"
    if positive_rate >= 0.70 and false_accept_rate <= 0.10:
        return "A_STABLE_CANDIDATE"
    if positive_rate < 0.55 and false_accept_rate <= 0.10:
        return "C_NARROW_OR_LEGACY_NOISE_REVIEW"
    if positive_rate >= 0.70 and false_accept_rate > 0.15:
        return "D_BROAD_OR_OVERLAP_REVIEW"
    if positive_rate < 0.55 and false_accept_rate > 0.15:
        return "E_BOUNDARY_CONFLICT_REVIEW"
    return "B_MINOR_BOUNDARY_REVIEW"


def combine_positive_and_negative_assessments(
    positive_per_label_path: str | Path,
    negative_per_label_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Join positive compatibility and hard-negative exclusion into a screen."""
    positive = {
        str(row["label_id"]): row for row in _read_jsonl(positive_per_label_path)
    }
    negative = {
        str(row["label_id"]): row for row in _read_jsonl(negative_per_label_path)
    }
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label_id, pos in sorted(positive.items()):
        neg = negative.get(label_id, {})
        positive_count = int(pos.get("planned") or pos.get("completed") or 0)
        positive_rate = pos.get("match_rate")
        positive_rate = float(positive_rate) if positive_rate is not None else None
        negative_count = int(neg.get("hard_negative_total") or 0)
        false_accept_rate = neg.get("false_accept_rate")
        false_accept_rate = (
            float(false_accept_rate) if false_accept_rate is not None else None
        )
        rows.append(
            {
                "label_id": label_id,
                "label_name": pos.get("label_name", ""),
                "positive_count": positive_count,
                "legacy_positive_match_rate": positive_rate,
                "sample_tier": pos.get("sample_tier"),
                "hard_negative_count": negative_count,
                "hard_negative_false_accept_rate": false_accept_rate,
                "final_screen": _combined_screen(
                    positive_count,
                    positive_rate,
                    negative_count,
                    false_accept_rate,
                ),
                "decision_warning": "This is an automatic screen, not authorization to rewrite a teacher definition.",
            }
        )
    with (output_dir / "label_assessments.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "labels": len(rows),
        "positive_labels": len(positive),
        "negative_labels": len(negative),
        "screen_counts": dict(
            sorted(Counter(row["final_screen"] for row in rows).items())
        ),
        "warning": "Positive rates measure compatibility with historical assignments; hard-negative rates measure sibling-boundary exclusion. Human review is required before definition edits.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
