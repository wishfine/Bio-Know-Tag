"""Export per-Label teacher review packages from positive and boundary audits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROBLEM_RISKS = {
    "P0_图谱冲突",
    "P0_明显异常",
    "P1_重点核验",
    "L0_极端长尾",
    "L1_长尾异常",
}


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row in {path} must be an object")
                yield value


def _positive_risk(metric: dict[str, Any], strategy: dict[str, Any] | None) -> str:
    """Reproduce the original 135-Label positive-review selection exactly."""
    planned = int(metric["planned"])
    match_rate = float(metric["match_rate"])
    zero_rate = float(metric["zero_rate"])
    final_strategy = (strategy or {}).get("final_strategy") or {}
    if final_strategy.get("status") == "taxonomy_hold":
        return "P0_图谱冲突"
    if planned < 30:
        return "L0_极端长尾"
    if planned < 300:
        if match_rate < 0.40 or zero_rate >= 0.50:
            return "L1_长尾异常"
        return "L2_长尾待核"
    if match_rate < 0.20 or zero_rate >= 0.60:
        return "P0_明显异常"
    if match_rate < 0.55 or zero_rate >= 0.30:
        return "P1_重点核验"
    if match_rate < 0.70 or metric["preliminary_grade"] != "A_STABLE_CANDIDATE":
        return "P2_边界观察"
    if bool(final_strategy.get("manual_followup_required")):
        return "P2_边界观察"
    return "S_正样本稳定"


def _manual_followup(strategy: dict[str, Any] | None) -> bool:
    return bool(
        (((strategy or {}).get("final_strategy") or {}).get(
            "manual_followup_required"
        ))
    )


def _score_band(score: float) -> str:
    if score < 0.10:
        return "basically_irrelevant_<0.10"
    if score < 0.70:
        return "related_below_definition_0.10_0.69"
    if score < 0.80:
        return "matched_0.70_0.79"
    return "high_confidence_>=0.80"


def _stable_hash(*values: Any) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()


def _quantile_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select deterministic score-quantile representatives, including both ends."""
    if len(rows) <= limit:
        return sorted(
            rows,
            key=lambda row: (
                float(row["relevance_score"]),
                _stable_hash(row["question_id"]),
            ),
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["relevance_score"]),
            _stable_hash(row["question_id"]),
        ),
    )
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indexes]


def _allocate_strata(counts: list[int], limit: int) -> list[int]:
    total = sum(counts)
    if total <= limit:
        return counts[:]
    allocation = [0] * len(counts)
    nonempty = [index for index, count in enumerate(counts) if count]
    if limit >= len(nonempty):
        for index in nonempty:
            allocation[index] = 1
    remaining = limit - sum(allocation)
    while remaining:
        candidates = [
            index for index, count in enumerate(counts) if allocation[index] < count
        ]
        index = max(
            candidates,
            key=lambda value: (
                counts[value] / total - allocation[value] / limit,
                counts[value],
                -value,
            ),
        )
        allocation[index] += 1
        remaining -= 1
    return allocation


def _stratified_positive_sample(
    rows: list[dict[str, Any]], *, matched: bool, limit: int
) -> list[dict[str, Any]]:
    if matched:
        strata = [
            [row for row in rows if 0.70 <= float(row["relevance_score"]) < 0.80],
            [row for row in rows if float(row["relevance_score"]) >= 0.80],
        ]
    else:
        strata = [
            [row for row in rows if float(row["relevance_score"]) < 0.10],
            [row for row in rows if 0.10 <= float(row["relevance_score"]) < 0.70],
        ]
    allocation = _allocate_strata([len(stratum) for stratum in strata], limit)
    chosen = []
    for stratum, count in zip(strata, allocation):
        chosen.extend(_quantile_sample(stratum, count))
    chosen.sort(
        key=lambda row: (
            float(row["relevance_score"]),
            str(row["question_id"]),
        )
    )
    return chosen


def _question_reference(
    row: dict[str, Any],
    task: dict[str, Any],
    image_context: dict[str, Any],
    label: dict[str, Any],
) -> dict[str, Any]:
    score = float(row["relevance_score"])
    question_id = str(row["question_id"])
    return {
        "label_name": str(label.get("label_name") or ""),
        "label_path": str(label.get("label_path") or ""),
        "question_id": question_id,
        "parent_id": str(
            image_context.get("parent_id")
            or task.get("parent_id")
            or question_id
        ),
        "question_type": str(task.get("unit_type") or "unknown"),
        "model_match": bool(row.get("match", score >= 0.70)),
        "model_score": score,
        "stem": str(task.get("stem") or ""),
        "stem_image_url": str(image_context.get("stem_image_url") or ""),
        "analysis_image_url": str(image_context.get("analysis_image_url") or ""),
        "task_id": str(row["task_id"]),
        "score_band": _score_band(score),
    }


def _positive_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    irrelevant = sum(float(row["relevance_score"]) < 0.10 for row in rows)
    boundary = sum(
        0.10 <= float(row["relevance_score"]) < 0.70 for row in rows
    )
    matched = sum(float(row["relevance_score"]) >= 0.70 for row in rows)
    if irrelevant + boundary + matched != total:
        raise ValueError("positive score bands do not sum to total")
    return {
        "total_questions": total,
        "basically_irrelevant_count": irrelevant,
        "basically_irrelevant_over_total": f"{irrelevant}/{total}",
        "basically_irrelevant_rate": round(irrelevant / total, 6) if total else None,
        "related_below_definition_count": boundary,
        "related_below_definition_over_total": f"{boundary}/{total}",
        "related_below_definition_rate": round(boundary / total, 6) if total else None,
        "matched_count": matched,
        "matched_over_total": f"{matched}/{total}",
        "matched_rate": round(matched / total, 6) if total else None,
        "ds_false_count": irrelevant + boundary,
        "ds_true_count": matched,
    }


def _safe_filename(index: int, label: dict[str, Any]) -> str:
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", str(label["label_name"]))
    name = name.strip("_")[:80] or "label"
    return f"{index:03d}_{label['label_id']}_{name}.json"


def _round_robin_source_sample(
    rows: list[dict[str, Any]], samples: dict[str, dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = samples[str(row["task_id"])]
        key = tuple(str(value) for value in sample.get("source_label_ids") or [])
        groups[key].append(row)
    for key in groups:
        groups[key].sort(
            key=lambda row: _stable_hash(row["question_id"], row["decision"])
        )
    chosen = []
    positions = defaultdict(int)
    keys = sorted(groups)
    while len(chosen) < limit:
        progressed = False
        for key in keys:
            position = positions[key]
            if position >= len(groups[key]):
                continue
            chosen.append(groups[key][position])
            positions[key] += 1
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return chosen


def _boundary_reference(
    row: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    first_stage: dict[str, dict[str, Any]],
    image_contexts: dict[str, dict[str, Any]],
    target_label: dict[str, Any],
) -> dict[str, Any]:
    sample = samples[str(row["task_id"])]
    first = first_stage[str(row["task_id"])]
    question_id = str(row["question_id"])
    context = image_contexts.get(question_id, {})
    return {
        "label_name": str(target_label.get("label_name") or ""),
        "label_path": str(target_label.get("label_path") or ""),
        "question_id": question_id,
        "parent_id": str(
            context.get("parent_id")
            or sample.get("parent_id")
            or question_id
        ),
        "question_type": str(sample.get("unit_type") or "unknown"),
        "model_match": bool(first.get("match", True)),
        "model_score": float(first["relevance_score"]),
        "stem": str(sample.get("stem") or ""),
        "stem_image_url": str(context.get("stem_image_url") or ""),
        "analysis_image_url": str(context.get("analysis_image_url") or ""),
        "task_id": str(row["task_id"]),
        "source_label_ids": [str(value) for value in sample.get("source_label_ids") or []],
        "source_label_names": list(sample.get("source_label_names") or []),
        "first_stage_target_score": float(first["relevance_score"]),
        "co_label_decision": str(row["decision"]),
    }


def export_teacher_review_packages(
    *,
    labels_path: str | Path,
    strategies_path: str | Path,
    positive_tasks_path: str | Path,
    positive_results_path: str | Path,
    positive_per_label_path: str | Path,
    image_context_path: str | Path,
    corrected_assessments_path: str | Path,
    hard_negative_samples_path: str | Path,
    hard_negative_results_path: str | Path,
    colabel_results_path: str | Path,
    output_dir: str | Path,
    examples_per_side: int = 10,
    high_boundary_threshold: float = 0.15,
    min_positive_count: int = 300,
    min_valid_negative_count: int = 20,
) -> dict[str, Any]:
    """Create two sibling directories with one compact JSON file per Label."""
    if examples_per_side < 1:
        raise ValueError("examples_per_side must be positive")
    labels = {str(row["label_id"]): row for row in _read_jsonl(labels_path)}
    strategies = {
        str(row["label_id"]): row for row in _read_jsonl(strategies_path)
    }
    metrics = {
        str(row["label_id"]): row for row in _read_jsonl(positive_per_label_path)
    }
    positive_tasks = {
        str(row["pair_id"]): row for row in _read_jsonl(positive_tasks_path)
    }
    image_contexts = {
        str(row["question_id"]): row for row in _read_jsonl(image_context_path)
    }
    corrected = {
        str(row["label_id"]): row for row in _read_jsonl(corrected_assessments_path)
    }
    positive_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(positive_results_path):
        positive_by_label[str(row["label_id"])].append(row)
    hard_samples = {
        str(row["pair_id"]): row for row in _read_jsonl(hard_negative_samples_path)
    }
    hard_results = {
        str(row["task_id"]): row for row in _read_jsonl(hard_negative_results_path)
    }
    colabel_results = {
        str(row["task_id"]): row for row in _read_jsonl(colabel_results_path)
    }
    if set(labels) != set(metrics) or set(labels) != set(corrected):
        raise ValueError("labels, positive metrics, and corrected assessments differ")
    positive_result_ids = {
        str(row["task_id"])
        for rows in positive_by_label.values()
        for row in rows
    }
    if positive_result_ids != set(positive_tasks):
        raise ValueError("positive tasks/results task IDs differ")

    old_problem_ids = set()
    risks = {}
    for label_id, metric in metrics.items():
        risk = _positive_risk(metric, strategies.get(label_id))
        risks[label_id] = risk
        if risk in PROBLEM_RISKS or _manual_followup(strategies.get(label_id)):
            old_problem_ids.add(label_id)

    additional_ids = {
        label_id
        for label_id, row in corrected.items()
        if label_id not in old_problem_ids
        and int(row["positive_count"]) >= min_positive_count
        and int(row["valid_negative_count"]) >= min_valid_negative_count
        and row.get("corrected_boundary_error_rate") is not None
        and float(row["corrected_boundary_error_rate"]) > high_boundary_threshold
    }

    root = Path(output_dir)
    positive_dir = root / "positive_issue_135"
    boundary_dir = root / "boundary_risk_additional"
    positive_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    def base_payload(label_id: str, package_kind: str) -> dict[str, Any]:
        label = labels[label_id]
        rows = positive_by_label[label_id]
        false_rows = [row for row in rows if float(row["relevance_score"]) < 0.70]
        true_rows = [row for row in rows if float(row["relevance_score"]) >= 0.70]
        return {
            "schema_version": "biology-teacher-review-v1",
            "package_kind": package_kind,
            "label": {
                key: label.get(key, "")
                for key in (
                    "label_id",
                    "label_name",
                    "label_path",
                    "label_type",
                    "definition",
                    "core_concepts",
                    "common_assessments",
                    "distinctions",
                )
            },
            "positive_risk_level": risks[label_id],
            "positive_coverage": _positive_summary(rows),
            "representative_positive_ds_false": [
                _question_reference(
                    row,
                    positive_tasks[str(row["task_id"])],
                    image_contexts.get(str(row["question_id"]), {}),
                    label,
                )
                for row in _stratified_positive_sample(
                    false_rows, matched=False, limit=examples_per_side
                )
            ],
            "representative_positive_ds_true": [
                _question_reference(
                    row,
                    positive_tasks[str(row["task_id"])],
                    image_contexts.get(str(row["question_id"]), {}),
                    label,
                )
                for row in _stratified_positive_sample(
                    true_rows, matched=True, limit=examples_per_side
                )
            ],
            "sampling_note": (
                "Positive examples are deterministic score-stratified representatives; "
                "fewer than the requested count means the Label has fewer available rows."
            ),
        }

    for index, label_id in enumerate(
        sorted(old_problem_ids, key=lambda value: labels[value]["label_name"]), 1
    ):
        payload = base_payload(label_id, "positive_issue_135")
        path = positive_dir / _safe_filename(index, labels[label_id])
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    colabel_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id, row in colabel_results.items():
        sample = hard_samples.get(task_id)
        if sample is None:
            raise ValueError(f"missing hard-negative sample for {task_id}")
        colabel_by_target[str(sample["label_id"])].append(row)
    hard_rejected_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id, result in hard_results.items():
        sample = hard_samples.get(task_id)
        if sample is not None and float(result["relevance_score"]) < 0.70:
            hard_rejected_by_target[str(sample["label_id"])].append(result)

    for index, label_id in enumerate(
        sorted(additional_ids, key=lambda value: labels[value]["label_name"]), 1
    ):
        payload = base_payload(label_id, "boundary_risk_additional")
        assessment = corrected[label_id]
        colabel_rows = colabel_by_target.get(label_id, [])

        def decision_rows(*decisions: str) -> list[dict[str, Any]]:
            wanted = set(decisions)
            return [row for row in colabel_rows if str(row["decision"]) in wanted]

        payload["boundary_selection_rule"] = {
            "excluded_from_positive_issue_135": True,
            "minimum_positive_count": min_positive_count,
            "minimum_valid_negative_count": min_valid_negative_count,
            "corrected_boundary_error_rate_must_exceed": high_boundary_threshold,
        }
        payload["corrected_boundary_summary"] = {
            key: assessment.get(key)
            for key in (
                "hard_negative_total",
                "first_stage_accepted",
                "first_stage_rejected",
                "colabel_decision_counts",
                "invalid_sibling_negatives",
                "unresolved",
                "valid_negative_count",
                "true_boundary_errors",
                "corrected_boundary_error_rate",
                "final_screen",
            )
        }
        payload["representative_target_excluded_after_colabel"] = [
            _boundary_reference(
                row, hard_samples, hard_results, image_contexts, labels[label_id]
            )
            for row in _round_robin_source_sample(
                decision_rows("目标Label边界过宽"), hard_samples, examples_per_side
            )
        ]
        payload["representative_reasonable_colabel"] = [
            _boundary_reference(
                row, hard_samples, hard_results, image_contexts, labels[label_id]
            )
            for row in _round_robin_source_sample(
                decision_rows("合理共标"), hard_samples, examples_per_side
            )
        ]
        payload["representative_source_label_suspect"] = [
            _boundary_reference(
                row, hard_samples, hard_results, image_contexts, labels[label_id]
            )
            for row in _round_robin_source_sample(
                decision_rows("来源Label不足以描述该题", "两侧Label均不充分"),
                hard_samples,
                examples_per_side,
            )
        ]
        rejected = _quantile_sample(
            hard_rejected_by_target.get(label_id, []), examples_per_side
        )
        rejected_references = []
        for row in rejected:
            task_id = str(row["task_id"])
            sample = hard_samples[task_id]
            question_id = str(row["question_id"])
            context = image_contexts.get(question_id, {})
            rejected_references.append(
                {
                    "label_name": str(labels[label_id].get("label_name") or ""),
                    "label_path": str(labels[label_id].get("label_path") or ""),
                    "question_id": question_id,
                    "parent_id": str(
                        context.get("parent_id")
                        or sample.get("parent_id")
                        or question_id
                    ),
                    "question_type": str(sample.get("unit_type") or "unknown"),
                    "model_match": bool(row.get("match", False)),
                    "model_score": float(row["relevance_score"]),
                    "stem": str(sample.get("stem") or ""),
                    "stem_image_url": str(context.get("stem_image_url") or ""),
                    "analysis_image_url": str(
                        context.get("analysis_image_url") or ""
                    ),
                    "task_id": task_id,
                    "source_label_ids": [
                        str(value) for value in sample.get("source_label_ids") or []
                    ],
                    "source_label_names": list(
                        sample.get("source_label_names") or []
                    ),
                }
            )
        payload["representative_first_stage_target_rejected"] = rejected_references
        path = boundary_dir / _safe_filename(index, labels[label_id])
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "schema_version": "biology-teacher-review-v1",
        "positive_issue_directory": str(positive_dir),
        "positive_issue_labels": len(old_problem_ids),
        "additional_boundary_directory": str(boundary_dir),
        "additional_boundary_labels": len(additional_ids),
        "examples_per_side": examples_per_side,
        "additional_selection": {
            "excluded_from_positive_issue_135": True,
            "minimum_positive_count": min_positive_count,
            "minimum_valid_negative_count": min_valid_negative_count,
            "corrected_boundary_error_rate_must_exceed": high_boundary_threshold,
        },
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
