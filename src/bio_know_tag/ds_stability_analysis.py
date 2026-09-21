"""Detailed post-hoc analysis for the DS stability experiment.

The stability runner stores every API choice, but its top-level report only
contains agreement rates.  This module turns those choices into an audit that
answers the practical question: did a condition select more Labels, fewer
Labels, or replace Labels relative to the baseline condition?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bio_know_tag.ds_stability import jaccard, strict_majority_ids


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _load_labels(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row["label_id"]): row
        for row in _read_jsonl(path)
        if row.get("label_id")
    }


def _load_latest_success(path: Path) -> dict[str, dict[str, Any]]:
    """Read append-only condition output, retaining the latest success."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id") or "")
        if question_id and not row.get("error"):
            rows[question_id] = row
    return rows


def _choice_sets(row: dict[str, Any]) -> list[set[str]]:
    values: list[set[str]] = []
    for choice in row.get("choices") or []:
        if choice.get("parse_error"):
            continue
        values.append({str(value) for value in choice.get("selected_label_ids") or []})
    return values


def _majority(row: dict[str, Any]) -> tuple[set[str], list[set[str]]]:
    choices = _choice_sets(row)
    return strict_majority_ids(choices), choices


def _card(label_id: str, labels: dict[str, dict[str, Any]]) -> dict[str, str]:
    label = labels.get(label_id, {})
    return {
        "label_id": label_id,
        "label_name": str(label.get("label_name") or ""),
        "label_path": str(label.get("label_path") or ""),
    }


def _comparison(
    baseline: set[str], condition: set[str]
) -> tuple[str, set[str], set[str]]:
    added = condition - baseline
    removed = baseline - condition
    if not added and not removed:
        return ("both_empty" if not baseline else "same_set", added, removed)
    delta = len(condition) - len(baseline)
    if delta > 0:
        category = "more_labels_with_replacements" if removed else "more_labels"
    elif delta < 0:
        category = "fewer_labels_with_replacements" if added else "fewer_labels"
    else:
        category = "same_count_replaced"
    return category, added, removed


def _new_bucket() -> dict[str, Any]:
    return {
        "questions": 0,
        "same_set": 0,
        "different_output": 0,
        "both_empty": 0,
        "more_labels": 0,
        "fewer_labels": 0,
        "same_count_replaced": 0,
        "more_labels_with_replacements": 0,
        "fewer_labels_with_replacements": 0,
        "baseline_empty_to_nonempty": 0,
        "baseline_nonempty_to_empty": 0,
        "baseline_assignments": 0,
        "condition_assignments": 0,
        "added_assignments": 0,
        "removed_assignments": 0,
        "net_assignment_delta": 0,
        "mean_baseline_selected_count": None,
        "mean_condition_selected_count": None,
        "mean_count_delta": None,
        "mean_selection_jaccard": None,
    }


def _finish_bucket(bucket: dict[str, Any], deltas: list[int], jaccards: list[float]) -> None:
    total = int(bucket["questions"])
    if total:
        bucket["mean_baseline_selected_count"] = round(
            bucket.pop("_baseline_count_sum", 0) / total, 6
        )
        bucket["mean_condition_selected_count"] = round(
            bucket.pop("_condition_count_sum", 0) / total, 6
        )
        bucket["mean_count_delta"] = round(sum(deltas) / total, 6)
        bucket["mean_selection_jaccard"] = round(sum(jaccards) / total, 6)
    else:
        bucket.pop("_baseline_count_sum", None)
        bucket.pop("_condition_count_sum", None)


def _summarize_pair(
    baseline_rows: dict[str, dict[str, Any]],
    condition_rows: dict[str, dict[str, Any]],
    groups: dict[str, str],
    labels: dict[str, dict[str, Any]],
    *,
    include_question_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    common = sorted(set(baseline_rows) & set(condition_rows))
    only_baseline = sorted(set(baseline_rows) - set(condition_rows))
    only_condition = sorted(set(condition_rows) - set(baseline_rows))
    overall = _new_bucket()
    by_group: dict[str, dict[str, Any]] = {}
    deltas: list[int] = []
    jaccards: list[float] = []
    question_rows: list[dict[str, Any]] = []

    for question_id in common:
        baseline_ids, baseline_choices = _majority(baseline_rows[question_id])
        condition_ids, condition_choices = _majority(condition_rows[question_id])
        category, added, removed = _comparison(baseline_ids, condition_ids)
        group = groups.get(question_id, "unknown")
        bucket = by_group.setdefault(group, _new_bucket())
        for target in (overall, bucket):
            target["questions"] += 1
            target["different_output"] += int(bool(added or removed))
            target["same_set"] += int(not added and not removed)
            target[category] += 1
            target["baseline_empty_to_nonempty"] += int(not baseline_ids and bool(condition_ids))
            target["baseline_nonempty_to_empty"] += int(bool(baseline_ids) and not condition_ids)
            target["baseline_assignments"] += len(baseline_ids)
            target["condition_assignments"] += len(condition_ids)
            target["added_assignments"] += len(added)
            target["removed_assignments"] += len(removed)
            target["net_assignment_delta"] += len(condition_ids) - len(baseline_ids)
            target.setdefault("_baseline_count_sum", 0)
            target.setdefault("_condition_count_sum", 0)
            target["_baseline_count_sum"] += len(baseline_ids)
            target["_condition_count_sum"] += len(condition_ids)
        delta = len(condition_ids) - len(baseline_ids)
        deltas.append(delta)
        score = jaccard(baseline_ids, condition_ids)
        jaccards.append(score)

        if include_question_rows:
            question_rows.append(
                {
                    "question_id": question_id,
                    "perturbation_group": group,
                    "comparison": category,
                    "baseline_choice_count": len(baseline_choices),
                    "condition_choice_count": len(condition_choices),
                    "baseline_choice_selected_counts": sorted(
                        len(value) for value in baseline_choices
                    ),
                    "condition_choice_selected_counts": sorted(
                        len(value) for value in condition_choices
                    ),
                    "baseline_selected_label_ids": sorted(baseline_ids),
                    "condition_selected_label_ids": sorted(condition_ids),
                    "baseline_selected_labels": [_card(i, labels) for i in sorted(baseline_ids)],
                    "condition_selected_labels": [_card(i, labels) for i in sorted(condition_ids)],
                    "added_label_ids": sorted(added),
                    "removed_label_ids": sorted(removed),
                    "added_labels": [_card(i, labels) for i in sorted(added)],
                    "removed_labels": [_card(i, labels) for i in sorted(removed)],
                    "baseline_selected_count": len(baseline_ids),
                    "condition_selected_count": len(condition_ids),
                    "count_delta": delta,
                    "selection_jaccard": round(score, 6),
                }
            )

    _finish_bucket(overall, deltas, jaccards)
    for group, bucket in by_group.items():
        group_rows = [
            row for row in question_rows
            if row["perturbation_group"] == group
        ]
        group_deltas = [row["count_delta"] for row in group_rows]
        group_jaccards = [row["selection_jaccard"] for row in group_rows]
        _finish_bucket(bucket, group_deltas, group_jaccards)

    summary = {
        "baseline_success": len(baseline_rows),
        "condition_success": len(condition_rows),
        "common_success": len(common),
        "baseline_only": len(only_baseline),
        "condition_only": len(only_condition),
        "overall": overall,
        "by_perturbation_group": by_group,
    }
    return summary, question_rows, {
        "baseline_only_question_ids": only_baseline,
        "condition_only_question_ids": only_condition,
    }


def analyze_stability_run(
    run_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    baseline: str | None = None,
    output_dir: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Analyze completed or partially completed conditions in one run directory."""
    root = Path(run_dir)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = [str(item["name"]) for item in manifest.get("conditions") or []]
    if not conditions:
        raise ValueError("run manifest has no conditions")
    baseline_name = baseline or conditions[0]
    if baseline_name not in conditions:
        raise ValueError(f"baseline {baseline_name!r} is not a condition")
    labels = _load_labels(labels_path)
    rows_by_condition = {
        name: _load_latest_success(root / name / "responses.jsonl")
        for name in conditions
    }
    expected = int(manifest.get("input_count") or 0)
    completion = {
        name: {
            "success": len(rows),
            "expected": expected,
            "complete": expected == 0 or len(rows) >= expected,
        }
        for name, rows in rows_by_condition.items()
    }
    if require_complete and not all(item["complete"] for item in completion.values()):
        raise ValueError(f"stability run is incomplete: {completion}")

    groups: dict[str, str] = {}
    for name in conditions:
        for row in rows_by_condition[name].values():
            question_id = str(row.get("question_id") or "")
            group = str(row.get("perturbation_group") or "unknown")
            if question_id:
                previous = groups.setdefault(question_id, group)
                if previous != group:
                    raise ValueError(f"question {question_id} has conflicting groups")

    output = Path(output_dir) if output_dir else root / "detailed-analysis"
    output.mkdir(parents=True, exist_ok=True)
    baseline_rows = rows_by_condition[baseline_name]
    pair_reports: list[dict[str, Any]] = []
    all_question_rows: list[dict[str, Any]] = []
    missing: dict[str, Any] = {}
    for name in conditions:
        if name == baseline_name:
            continue
        summary, question_rows, missing_rows = _summarize_pair(
            baseline_rows,
            rows_by_condition[name],
            groups,
            labels,
            include_question_rows=True,
        )
        pair_report = {
            "baseline": baseline_name,
            "condition": name,
            **summary,
        }
        pair_reports.append(pair_report)
        all_question_rows.extend(
            {"condition": name, **row} for row in question_rows
        )
        missing[name] = missing_rows

    # A compact per-condition overview also includes the baseline itself.
    condition_overview = []
    for name in conditions:
        counts = [len(_majority(row)[0]) for row in rows_by_condition[name].values()]
        condition_overview.append(
            {
                "condition": name,
                "success": len(rows_by_condition[name]),
                "expected": expected,
                "mean_selected_count": round(sum(counts) / len(counts), 6) if counts else None,
                "empty_majority": sum(value == 0 for value in counts),
                "prompt_sha256_count": len({str(row.get("prompt_sha256") or "") for row in rows_by_condition[name].values()}),
            }
        )

    report = {
        "experiment": "ds-stability-detailed-analysis-v1",
        "run_dir": str(root),
        "baseline": baseline_name,
        "conditions": conditions,
        "completion": completion,
        "condition_overview": condition_overview,
        "pairwise_to_baseline": pair_reports,
        "missing_by_condition": missing,
        "note": "For n>1, condition_selected_label_ids are strict-majority selections; raw choices remain in each condition's responses.jsonl.",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_question.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(all_question_rows, key=lambda value: (value["condition"], value["question_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Label-level counts answer which Labels are systematically added or removed.
    per_label: list[dict[str, Any]] = []
    for pair in pair_reports:
        name = pair["condition"]
        rows = [row for row in all_question_rows if row["condition"] == name]
        label_ids = set(labels)
        for row in rows:
            label_ids.update(row["baseline_selected_label_ids"])
            label_ids.update(row["condition_selected_label_ids"])
        for label_id in sorted(label_ids):
            baseline_count = sum(label_id in row["baseline_selected_label_ids"] for row in rows)
            condition_count = sum(label_id in row["condition_selected_label_ids"] for row in rows)
            added_count = sum(label_id in row["added_label_ids"] for row in rows)
            removed_count = sum(label_id in row["removed_label_ids"] for row in rows)
            per_label.append(
                {
                    "condition": name,
                    **_card(label_id, labels),
                    "common_questions": len(rows),
                    "baseline_selected_count": baseline_count,
                    "condition_selected_count": condition_count,
                    "added_count": added_count,
                    "removed_count": removed_count,
                    "net_count_delta": condition_count - baseline_count,
                }
            )
    with (output / "per_label.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_label:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    _write_markdown(output / "summary.md", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DS 稳定性详细分析",
        "",
        f"基线条件：`{report['baseline']}`",
        "",
        "## 完成情况",
        "",
        "| 条件 | 成功题数 | 计划题数 | 是否完成 | 平均最终选中数 | 空选题数 | Prompt hash 数 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in report["condition_overview"]:
        lines.append(
            f"| `{row['condition']}` | {row['success']} | {row['expected']} | "
            f"{'是' if row['success'] >= row['expected'] else '否'} | "
            f"{row['mean_selected_count'] if row['mean_selected_count'] is not None else '-'} | "
            f"{row['empty_majority']} | {row['prompt_sha256_count']} |"
        )
    lines += ["", "## 相对基线：多选、少选还是替换", ""]
    lines += [
        "| 条件 | 共同题 | 输出不同 | 多选题数 | 少选题数 | 同数替换 | 新增Label数 | 移除Label数 | 净变化 | 平均选中数变化 | 平均Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in report["pairwise_to_baseline"]:
        stats = pair["overall"]
        difference_rate = (
            f"{stats['different_output'] / pair['common_success']:.2%}"
            if pair["common_success"]
            else "0.00%"
        )
        lines.append(
            f"| `{pair['condition']}` | {pair['common_success']} | {stats['different_output']} "
            f"({difference_rate}) | "
            f"{stats['more_labels'] + stats['more_labels_with_replacements']} | "
            f"{stats['fewer_labels'] + stats['fewer_labels_with_replacements']} | "
            f"{stats['same_count_replaced']} | {stats['added_assignments']} | {stats['removed_assignments']} | "
            f"{stats['net_assignment_delta']} | {stats['mean_count_delta']} | {stats['mean_selection_jaccard']} |"
        )
    lines += [
        "",
        "`多选题数/少选题数`按最终严格多数票的 Label 数量相对基线计算；同样数量但 Label 集合不同的题单列为`同数替换`。",
        "",
        "## 分扰动组统计",
        "",
    ]
    for pair in report["pairwise_to_baseline"]:
        lines += [f"### `{pair['condition']}`", "", "| 组 | 共同题 | 多选 | 少选 | 同数替换 | 净Label变化 |", "|---|---:|---:|---:|---:|---:|"]
        for group, stats in sorted(pair["by_perturbation_group"].items()):
            lines.append(
                f"| `{group}` | {stats['questions']} | {stats['more_labels'] + stats['more_labels_with_replacements']} | "
                f"{stats['fewer_labels'] + stats['fewer_labels_with_replacements']} | {stats['same_count_replaced']} | {stats['net_assignment_delta']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
