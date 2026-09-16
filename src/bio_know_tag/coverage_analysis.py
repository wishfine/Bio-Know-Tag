"""Analyze partial or complete definition-coverage results without calling DS."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
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


def _read_results_tolerating_partial_line(
    path: str | Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    malformed = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = str(row["task_id"])
                score = float(row["relevance_score"])
                if not 0 <= score <= 1:
                    raise ValueError("score out of range")
                latest[task_id] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed += 1
    return latest, malformed


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _score_band(score: float) -> str:
    if score >= 0.80:
        return "high_match"
    if score >= MATCH_THRESHOLD:
        return "borderline_match"
    if score >= 0.40:
        return "gray"
    if score == 0:
        return "zero"
    return "weak_related"


def _sample_tier(planned: int) -> str:
    if planned < 30:
        return "LT1_EXTREME_1_29"
    if planned < 100:
        return "LT2_SEVERE_30_99"
    if planned < 300:
        return "LT3_MODERATE_100_299"
    if planned < 500:
        return "ADEQUATE_300_499"
    return "CAPPED_500"


def _metric_row(
    label_id: str,
    label_name: str,
    planned: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = len(rows)
    scores = [float(row["relevance_score"]) for row in rows]
    matches = sum(score >= MATCH_THRESHOLD for score in scores)
    high = sum(score >= 0.80 for score in scores)
    zero = sum(score == 0 for score in scores)
    gray = sum(0.40 <= score < MATCH_THRESHOLD for score in scores)
    weak = sum(0 < score < 0.40 for score in scores)
    by_type: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        unit_type = str(row.get("question_type") or "unknown")
        by_type[unit_type].append(score)
    standalone = by_type.get("standalone", [])
    subquestion = by_type.get("sub_question", []) + by_type.get(
        "orphan_sub_question", []
    )
    standalone_rate = _safe_rate(
        sum(score >= MATCH_THRESHOLD for score in standalone), len(standalone)
    )
    subquestion_rate = _safe_rate(
        sum(score >= MATCH_THRESHOLD for score in subquestion), len(subquestion)
    )
    type_gap = (
        round(standalone_rate - subquestion_rate, 6)
        if standalone_rate is not None and subquestion_rate is not None
        else None
    )
    completion_rate = _safe_rate(completed, planned)
    match_rate = _safe_rate(matches, completed)
    zero_rate = _safe_rate(zero, completed)
    gray_rate = _safe_rate(gray, completed)
    issue_flags: list[str] = []
    if completed < planned:
        issue_flags.append("incomplete")
    if planned < 50:
        issue_flags.append("low_sample")
    if planned < 300:
        issue_flags.append("long_tail_under_300")
    if completed >= 30 and match_rate is not None and match_rate < 0.55:
        issue_flags.append("low_positive_coverage")
    if completed >= 30 and zero_rate is not None and zero_rate >= 0.30:
        issue_flags.append("high_zero_rate")
    if completed >= 30 and gray_rate is not None and gray_rate >= 0.15:
        issue_flags.append("boundary_gray")
    if (
        len(standalone) >= 10
        and len(subquestion) >= 10
        and type_gap is not None
        and type_gap >= 0.20
    ):
        issue_flags.append("parent_union_suspected")
    if completed >= 30 and match_rate is not None and match_rate >= 0.90:
        issue_flags.append("very_high_positive_acceptance")
    stable = (
        completed == planned
        and planned >= 50
        and match_rate is not None
        and match_rate >= 0.70
        and zero_rate is not None
        and zero_rate <= 0.15
        and gray_rate is not None
        and gray_rate <= 0.10
        and "parent_union_suspected" not in issue_flags
    )
    sample_tier = _sample_tier(planned)
    if completed == planned and planned < 300:
        grade = "U_LONG_TAIL_REVIEW"
    elif completed < planned:
        grade = "SNAPSHOT_INCOMPLETE"
    elif "parent_union_suspected" in issue_flags or "high_zero_rate" in issue_flags:
        grade = "D_NON_DEFINITION_ISSUE_SUSPECTED"
    elif "low_positive_coverage" in issue_flags or "boundary_gray" in issue_flags:
        grade = "C_NEEDS_DIAGNOSIS"
    elif stable:
        grade = "A_STABLE_CANDIDATE"
    else:
        grade = "B_MINOR_REVIEW"
    return {
        "label_id": label_id,
        "label_name": label_name,
        "sample_tier": sample_tier,
        "planned": planned,
        "completed": completed,
        "completion_rate": completion_rate,
        "match": matches,
        "match_rate": match_rate,
        "high_match": high,
        "high_match_rate": _safe_rate(high, completed),
        "zero": zero,
        "zero_rate": zero_rate,
        "gray": gray,
        "gray_rate": gray_rate,
        "weak_related": weak,
        "weak_related_rate": _safe_rate(weak, completed),
        "mean_score": round(sum(scores) / completed, 6) if completed else None,
        "standalone_completed": len(standalone),
        "standalone_match_rate": standalone_rate,
        "sub_question_completed": len(subquestion),
        "sub_question_match_rate": subquestion_rate,
        "standalone_minus_sub_question": type_gap,
        "issue_flags": issue_flags,
        "preliminary_grade": grade,
        "broadness_status": "not_tested_requires_hard_negatives",
    }


def _top_rows(
    rows: list[dict[str, Any]], key: str, *, reverse: bool = True, limit: int = 15
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get(key) is not None]
    return sorted(eligible, key=lambda row: (row[key], row["completed"]), reverse=reverse)[
        :limit
    ]


def _markdown_report(report: dict[str, Any], per_label: list[dict[str, Any]]) -> str:
    lines = [
        "# 生物Label释义覆盖中期快照",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 当前进度",
        "",
        f"- 已完成：{report['completed_tasks']:,} / {report['planned_tasks']:,}（{report['completion_rate']:.2%}）",
        f"- 已出现Label：{report['labels_with_results']} / {report['labels_planned']}",
        f"- 当前匹配率：{report['match_rate']:.2%}" if report["match_rate"] is not None else "- 当前匹配率：无数据",
        f"- 唯一题目数：{report['unique_questions']:,}",
        "",
        "> 这是运行中快照。未完成Label的比例和分档不能视为最终结论；当前只有历史正样本，不能据此认定释义太宽。",
        "",
        "## 初筛等级",
        "",
        "| 等级 | Label数 |",
        "|---|---:|",
    ]
    for grade, count in sorted(report["preliminary_grade_counts"].items()):
        lines.append(f"| {grade} | {count} |")
    lines.extend(
        [
            "",
            "## 样本量与长尾",
            "",
            f"- 少于300道独立题的长尾Label：{report['long_tail_labels_under_300']}个。",
            "- LT1：1–29题；LT2：30–99题；LT3：100–299题。长尾Label不做全自动释义结论，必须重点人工复核。",
            "",
            "## 当前低匹配率Label（仅展示已有结果）",
            "",
            "| Label | 完成/计划 | 匹配率 | 0分率 | 灰度率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in _top_rows(per_label, "match_rate", reverse=False):
        lines.append(
            f"| {row['label_name']} | {row['completed']}/{row['planned']} | "
            f"{row['match_rate']:.2%} | {row['zero_rate']:.2%} | {row['gray_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 当前父题并集污染候选",
            "",
            "| Label | 独立题匹配率 | 小题匹配率 | 差值 |",
            "|---|---:|---:|---:|",
        ]
    )
    parent_rows = [
        row for row in per_label if "parent_union_suspected" in row["issue_flags"]
    ]
    for row in sorted(
        parent_rows,
        key=lambda item: item["standalone_minus_sub_question"],
        reverse=True,
    )[:15]:
        lines.append(
            f"| {row['label_name']} | {row['standalone_match_rate']:.2%} | "
            f"{row['sub_question_match_rate']:.2%} | {row['standalone_minus_sub_question']:.2%} |"
        )
    if not parent_rows:
        lines.append("| 暂无满足初筛门槛的Label | - | - | - |")
    lines.extend(
        [
            "",
            "## 解读限制",
            "",
            "- `A_STABLE_CANDIDATE`只是正样本覆盖稳定候选，仍需硬负样本验证是否偏宽。",
            "- 低匹配率可能来自释义偏窄、旧标签错误、父题污染或数据问题，必须看抽样证据后定性。",
            "- 计划样本少于300的Label统一标记为长尾，按LT1/LT2/LT3分级重点复核，不自动修改释义。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_coverage_snapshot(
    tasks_path: str | Path,
    results_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    samples_per_band: int = 3,
) -> dict[str, Any]:
    """Analyze an append-only results snapshot and emit inspectable artifacts."""
    if samples_per_band < 1:
        raise ValueError("samples_per_band must be positive")
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        str(row["label_id"]): row for row in _read_jsonl(labels_path)
    }
    tasks = list(_read_jsonl(tasks_path))
    task_by_id = {str(task["pair_id"]): task for task in tasks}
    planned_by_label = Counter(str(task["label_id"]) for task in tasks)
    results, malformed = _read_results_tolerating_partial_line(results_path)
    usable_results = {
        task_id: row for task_id, row in results.items() if task_id in task_by_id
    }
    unknown_result_tasks = len(results) - len(usable_results)
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable_results.values():
        rows_by_label[str(row["label_id"])].append(row)
    per_label = [
        _metric_row(
            label_id,
            str(labels.get(label_id, {}).get("label_name") or ""),
            planned_by_label[label_id],
            rows_by_label.get(label_id, []),
        )
        for label_id in labels
        if planned_by_label[label_id]
    ]
    per_label.sort(key=lambda row: row["label_id"])
    with (output_dir / "per_label.jsonl").open("w", encoding="utf-8") as output:
        for row in per_label:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    long_tail_rows = [row for row in per_label if row["planned"] < 300]
    with (output_dir / "long_tail_labels.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in sorted(long_tail_rows, key=lambda item: (item["planned"], item["label_name"])):
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    review_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task_id, result in usable_results.items():
        task = task_by_id[task_id]
        label_id = str(result["label_id"])
        score = float(result["relevance_score"])
        band = _score_band(score)
        key = (label_id, band)
        if len(review_buckets[key]) < samples_per_band:
            review_buckets[key].append(
                {
                    "task_id": task_id,
                    "label_id": label_id,
                    "label_name": labels.get(label_id, {}).get("label_name", ""),
                    "question_id": result["question_id"],
                    "question_type": result.get("question_type"),
                    "score_band": band,
                    "relevance_score": score,
                    "match": score >= MATCH_THRESHOLD,
                    "parent_stem": task.get("parent_stem", ""),
                    "stem": task.get("stem", ""),
                    "options": task.get("options", ""),
                    "answer_text": task.get("answer_text", ""),
                    "analysis": task.get("analysis", ""),
                }
            )
    with (output_dir / "review_samples.jsonl").open("w", encoding="utf-8") as output:
        for key in sorted(review_buckets):
            for row in review_buckets[key]:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    scores = [float(row["relevance_score"]) for row in usable_results.values()]
    matches = sum(score >= MATCH_THRESHOLD for score in scores)
    question_scores: dict[str, list[float]] = defaultdict(list)
    for row in usable_results.values():
        question_scores[str(row["question_id"])].append(float(row["relevance_score"]))
    grade_counts: Counter[str] = Counter()
    for values in question_scores.values():
        best = max(values)
        if best >= 0.80:
            grade_counts["A_>=0.80"] += 1
        elif best >= MATCH_THRESHOLD:
            grade_counts["B_0.70-0.79"] += 1
        elif best >= 0.40:
            grade_counts["C_0.40-0.69"] += 1
        elif best == 0:
            grade_counts["D_zero"] += 1
        else:
            grade_counts["D_0.01-0.39"] += 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "planned_tasks": len(tasks),
        "completed_tasks": len(usable_results),
        "completion_rate": round(len(usable_results) / len(tasks), 6) if tasks else 0.0,
        "pending_tasks": len(tasks) - len(usable_results),
        "malformed_result_lines_ignored": malformed,
        "unknown_result_tasks_ignored": unknown_result_tasks,
        "labels_planned": len(planned_by_label),
        "labels_with_results": len(rows_by_label),
        "unique_questions": len(question_scores),
        "match": matches,
        "match_rate": _safe_rate(matches, len(scores)),
        "score_band_counts": dict(sorted(Counter(_score_band(score) for score in scores).items())),
        "question_training_grade_distribution": dict(sorted(grade_counts.items())),
        "preliminary_grade_counts": dict(
            sorted(Counter(row["preliminary_grade"] for row in per_label).items())
        ),
        "long_tail_labels_under_300": len(long_tail_rows),
        "sample_tier_counts": dict(
            sorted(Counter(row["sample_tier"] for row in per_label).items())
        ),
        "issue_flag_counts": dict(
            sorted(Counter(flag for row in per_label for flag in row["issue_flags"]).items())
        ),
        "broadness_warning": "Positive-only coverage cannot determine whether a definition is too broad; hard negatives are required.",
        "snapshot_warning": "This report may reflect an incomplete append-only run and is not a final taxonomy decision.",
    }
    (output_dir / "snapshot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "snapshot_report.md").write_text(
        _markdown_report(report, per_label), encoding="utf-8"
    )
    return report
