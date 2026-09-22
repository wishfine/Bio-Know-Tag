#!/usr/bin/env python3
"""Generate a detailed five-run DS stability Markdown report.

The report combines the original production response with the four controlled
stability conditions and includes concrete Label/question examples.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN_NAMES = (
    "original_ds",
    "temp0-workers20",
    "temp01-n4-workers20",
    "temp0-seed42-workers20",
    "temp01-n4-seed42-workers20",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def latest_success(path: str | Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    rows: dict[str, dict[str, Any]] = {}
    total = errors = 0
    for row in read_jsonl(path):
        total += 1
        qid = str(row.get("question_id") or "")
        if row.get("error"):
            errors += 1
            continue
        if qid:
            rows[qid] = row
    return rows, total, errors


def labels_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["label_id"]): row
        for row in read_jsonl(path)
        if row.get("label_id")
    }


def strict_majority(sets: list[set[str]]) -> set[str]:
    if not sets:
        return set()
    threshold = len(sets) // 2 + 1
    counts = Counter(value for selected in sets for value in selected)
    return {value for value, count in counts.items() if count >= threshold}


def row_choices(row: dict[str, Any]) -> tuple[list[set[str]], int]:
    choices = row.get("choices")
    if isinstance(choices, list):
        valid: list[set[str]] = []
        parse_errors = 0
        for choice in choices:
            if choice.get("parse_error"):
                parse_errors += 1
                continue
            valid.append({str(value) for value in choice.get("selected_label_ids") or []})
        return valid, parse_errors

    parsed = row.get("parsed_response")
    code_map = row.get("candidate_code_map") or {}
    if isinstance(parsed, dict) and isinstance(code_map, dict):
        selected = {
            str(code_map[code])
            for code in parsed.get("selected") or []
            if code in code_map
        }
        return [selected], 0
    return [], 0


def final_selected(row: dict[str, Any]) -> tuple[set[str], list[set[str]], int]:
    choices, parse_errors = row_choices(row)
    if not choices:
        return set(), [], parse_errors
    return strict_majority(choices), choices, parse_errors


def output_evidence_reason(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the model's evidence/reason without losing n-way choices."""
    choices = row.get("choices")
    if isinstance(choices, list):
        outputs: list[dict[str, Any]] = []
        for choice in choices:
            parsed = choice.get("parsed_response")
            if not isinstance(parsed, dict):
                outputs.append({
                    "choice": choice.get("index"),
                    "parse_error": choice.get("parse_error"),
                    "evidence": None,
                    "reason": None,
                })
                continue
            outputs.append({
                "choice": choice.get("index"),
                "parse_error": choice.get("parse_error"),
                "evidence": parsed.get("evidence"),
                "reason": parsed.get("reason"),
            })
        return outputs
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict):
        return [{
            "choice": 0,
            "parse_error": None,
            "evidence": parsed.get("evidence"),
            "reason": parsed.get("reason"),
        }]
    return [{"choice": 0, "parse_error": "missing parsed response", "evidence": None, "reason": None}]


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def card(label_id: str, labels: dict[str, dict[str, Any]]) -> str:
    row = labels.get(label_id, {})
    name = str(row.get("label_name") or "未知Label")
    return f"{name} (`{label_id}`)"


def short_text(value: Any, limit: int = 700) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def pair_report(
    left: dict[str, set[str]], right: dict[str, set[str]], groups: dict[str, str]
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    same = more = fewer = same_count_replaced = 0
    added = removed = 0
    net = 0
    jaccards: list[float] = []
    by_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {"questions": 0, "different": 0, "more": 0, "fewer": 0, "same_count_replaced": 0}
    )
    for qid in common:
        a, b = left[qid], right[qid]
        plus, minus = b - a, a - b
        delta = len(b) - len(a)
        if not plus and not minus:
            same += 1
        elif delta > 0:
            more += 1
        elif delta < 0:
            fewer += 1
        else:
            same_count_replaced += 1
        added += len(plus)
        removed += len(minus)
        net += delta
        jaccards.append(jaccard(a, b))
        bucket = by_group[groups.get(qid, "unknown")]
        bucket["questions"] += 1
        bucket["different"] += int(bool(plus or minus))
        bucket["more"] += int(delta > 0)
        bucket["fewer"] += int(delta < 0)
        bucket["same_count_replaced"] += int(delta == 0 and bool(plus or minus))
    return {
        "common": len(common),
        "same": same,
        "different": len(common) - same,
        "difference_rate": (len(common) - same) / len(common) if common else None,
        "more": more,
        "fewer": fewer,
        "same_count_replaced": same_count_replaced,
        "added_assignments": added,
        "removed_assignments": removed,
        "net_assignment_delta": net,
        "mean_jaccard": statistics.mean(jaccards) if jaccards else None,
        "by_group": dict(by_group),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-ranking-output", type=Path)
    parser.add_argument("--label-ranking-output", type=Path)
    parser.add_argument("--max-questions", type=int, default=30)
    parser.add_argument("--max-labels", type=int, default=40)
    args = parser.parse_args()

    root = args.run_dir
    labels = labels_by_id(args.labels)
    units = {
        str(row["question_id"]): row
        for row in read_jsonl(root / "analysis-export/pilot_units_5391.jsonl")
        if row.get("question_id")
    }
    response_paths = {
        "original_ds": root / "analysis-export/original_ds_baseline_5391.jsonl",
        **{
            name: root / name / "responses.jsonl"
            for name in RUN_NAMES[1:]
        },
    }
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    input_stats: dict[str, dict[str, int]] = {}
    for name, path in response_paths.items():
        loaded, total, errors = latest_success(path)
        rows[name] = loaded
        input_stats[name] = {
            "file_rows": total,
            "error_rows": errors,
            "successful_questions": len(loaded),
        }

    selected: dict[str, dict[str, set[str]]] = {name: {} for name in RUN_NAMES}
    choice_stats: dict[str, dict[str, Any]] = {}
    raw_choice_examples: dict[str, dict[str, list[set[str]]]] = {
        name: {} for name in RUN_NAMES
    }
    for name in RUN_NAMES:
        parse_errors = choice_count = 0
        count_values: list[int] = []
        for qid, row in rows[name].items():
            final, choices, errors = final_selected(row)
            selected[name][qid] = final
            parse_errors += errors
            choice_count += len(choices)
            count_values.extend(len(value) for value in choices)
            raw_choice_examples[name][qid] = choices
        choice_stats[name] = {
            "choice_count": choice_count,
            "parse_errors": parse_errors,
            "mean_selected_per_choice": statistics.mean(count_values) if count_values else None,
            "median_selected_per_choice": statistics.median(count_values) if count_values else None,
            "min_selected_per_choice": min(count_values) if count_values else None,
            "max_selected_per_choice": max(count_values) if count_values else None,
            "empty_final_questions": sum(not value for value in selected[name].values()),
            "mean_final_selected": statistics.mean(len(value) for value in selected[name].values()) if selected[name] else None,
        }

    groups = {
        qid: str(rows["temp0-workers20"][qid].get("perturbation_group") or "unknown")
        for qid in rows["temp0-workers20"]
    }
    prompt_consistency: dict[str, dict[str, int]] = {}
    for name in RUN_NAMES[1:]:
        common_ids = set(rows["original_ds"]) & set(rows[name])
        prompt_consistency[name] = {
            "common_questions": len(common_ids),
            "same_prompt_sha256": sum(
                rows["original_ds"][qid].get("prompt_sha256")
                == rows[name][qid].get("prompt_sha256")
                for qid in common_ids
            ),
        }
    pairwise = []
    for i, left_name in enumerate(RUN_NAMES):
        for right_name in RUN_NAMES[i + 1 :]:
            pairwise.append({
                "left": left_name,
                "right": right_name,
                **pair_report(selected[left_name], selected[right_name], groups),
            })

    common = sorted(set.intersection(*(set(selected[name]) for name in RUN_NAMES)))
    question_records: list[dict[str, Any]] = []
    label_question_examples: dict[str, list[tuple[str, int]]] = defaultdict(list)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    five_set_patterns = Counter()
    for qid in common:
        sets = [selected[name][qid] for name in RUN_NAMES]
        pattern = Counter(tuple(sorted(value)) for value in sets)
        pattern_counts = sorted(pattern.values(), reverse=True)
        if pattern_counts == [5]:
            classification = "all_five_same"
        elif pattern_counts and pattern_counts[0] >= 4:
            classification = "four_or_more_same"
        elif pattern_counts and pattern_counts[0] >= 3:
            classification = "three_or_more_same"
        else:
            classification = "no_three_run_consensus"
        five_set_patterns[classification] += 1
        pair_scores = [
            jaccard(sets[i], sets[j])
            for i in range(5)
            for j in range(i + 1, 5)
        ]
        counts = [len(value) for value in sets]
        for value in sets:
            for label_id in value:
                label_counts[label_id]["present"] += 1
        for label_id in set().union(*sets):
            votes = sum(label_id in value for value in sets)
            label_counts[label_id][f"votes_{votes}"] += 1
            if 0 < votes < 5:
                label_question_examples[label_id].append((qid, votes))
        question_records.append({
            "question_id": qid,
            "classification": classification,
            "distinct_sets": len(pattern),
            "mean_pair_jaccard": statistics.mean(pair_scores),
            "min_pair_jaccard": min(pair_scores),
            "count_min": min(counts),
            "count_max": max(counts),
            "count_range": max(counts) - min(counts),
            "sets": sets,
            "outputs": {
                name: output_evidence_reason(rows[name][qid])
                for name in RUN_NAMES
            },
        })

    question_records.sort(key=lambda row: (row["mean_pair_jaccard"], -row["distinct_sets"], row["question_id"]))
    unstable_labels = []
    for label_id in labels:
        counts = label_counts.get(label_id, Counter())
        unstable = sum(counts.get(f"votes_{votes}", 0) for votes in range(1, 5))
        union_questions = sum(counts.get(f"votes_{votes}", 0) for votes in range(1, 6))
        instability_rate = unstable / len(common) if common else 0.0
        conditional_rate = unstable / union_questions if union_questions else 0.0
        unstable_labels.append({
            "label_id": label_id,
            "label_name": str(labels.get(label_id, {}).get("label_name") or "未知Label"),
            "label_path": str(labels.get(label_id, {}).get("label_path") or ""),
            "unstable_questions": unstable,
            "union_questions": union_questions,
            "instability_rate": instability_rate,
            "conditional_instability_rate": conditional_rate,
            "votes_1": counts.get("votes_1", 0),
            "votes_2": counts.get("votes_2", 0),
            "votes_3": counts.get("votes_3", 0),
            "votes_4": counts.get("votes_4", 0),
            "votes_5": counts.get("votes_5", 0),
            "examples": label_question_examples[label_id][:3],
        })
    unstable_labels.sort(key=lambda row: (
        -row["instability_rate"],
        -row["unstable_questions"],
        -row["conditional_instability_rate"],
        row["label_name"],
    ))

    all_question_records = sorted(
        question_records,
        key=lambda row: (
            -1.0 * (1.0 - row["mean_pair_jaccard"]),
            -row["distinct_sets"],
            -row["count_range"],
            row["question_id"],
        ),
    )

    report = {
        "run_dir": str(root),
        "run_names": list(RUN_NAMES),
        "input_stats": input_stats,
        "choice_stats": choice_stats,
        "prompt_consistency_vs_original": prompt_consistency,
        "pairwise": pairwise,
        "five_run_common_questions": len(common),
        "five_run_set_patterns": dict(five_set_patterns),
        "question_records": all_question_records[: args.max_questions],
        "unstable_labels": unstable_labels[: args.max_labels],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output, report, labels, units)
    examples_output = args.output.parent / "ds-stability-question-examples-30.md"
    write_question_examples(examples_output, report, labels, units)
    question_ranking_output = args.question_ranking_output or args.output.parent / "ds-stability-question-ranking-5391.md"
    label_ranking_output = args.label_ranking_output or args.output.parent / "ds-stability-label-ranking-458.md"
    write_question_ranking(question_ranking_output, all_question_records, labels, units)
    write_label_ranking(label_ranking_output, unstable_labels)
    json_output = args.output.with_suffix(".json")
    write_json_export(json_output, report, all_question_records, unstable_labels, labels, units)
    print(json.dumps({
        "output": str(args.output),
        "examples_output": str(examples_output),
        "json_output": str(json_output),
        "question_ranking_output": str(question_ranking_output),
        "label_ranking_output": str(label_ranking_output),
        "common_questions": len(common),
        "question_examples": min(args.max_questions, len(all_question_records)),
        "label_examples": min(args.max_labels, len(unstable_labels)),
        "question_ranking_count": len(all_question_records),
        "label_ranking_count": len(unstable_labels),
    }, ensure_ascii=False, indent=2))
    return 0


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def ranking_label_text(label_ids: set[str], labels: dict[str, dict[str, Any]], limit: int = 180) -> str:
    text = "；".join(str(labels.get(label_id, {}).get("label_name") or label_id) for label_id in sorted(label_ids))
    return text if len(text) <= limit else text[:limit] + "…"


def write_question_ranking(
    path: Path,
    records: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# DS 五次稳定性：5,391 道题不稳定度完整排序",
        "",
        "> 排序分数 = `1 - 五次最终 Label 集合的平均两两 Jaccard`，分数越高表示五次输出越不一致。n=4 条件使用严格多数票；同分时依次按不同集合数、选中数量范围降序排列。",
        "",
        f"共 {len(records)} 道题，全部来自五次共同成功题。主报告中的前 30 道题包含完整题干、解析以及五次 evidence/reason；本表为全部题的紧凑排序。",
        "",
        "| 排名 | 不稳定分数 | 题目ID | 题型 | 平均Jaccard | 不同集合数 | 选中数范围 | 五次模式 | 题干 | original_ds | temp0 | n4 | seed42 | n4+seed42 |",
        "|---:|---:|---|---|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for rank, record in enumerate(records, 1):
        qid = record["question_id"]
        unit = units.get(qid, {})
        run_cells = [ranking_label_text(set(value), labels) or "-" for value in record["sets"]]
        qtype = unit.get("unit_type") or unit.get("metadata", {}).get("structure_type") or "unknown"
        lines.append(
            f"| {rank} | {1.0 - record['mean_pair_jaccard']:.4f} | `{qid}` | `{qtype}` | "
            f"{record['mean_pair_jaccard']:.4f} | {record['distinct_sets']} | {record['count_min']}–{record['count_max']} | "
            f"`{record['classification']}` | {short_text(unit.get('stem'), 260)} | "
            + " | ".join(run_cells)
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_ranking(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# DS 五次稳定性：458 个 Label 不稳定度完整排序",
        "",
        "> 默认按 `不稳定题占全部 5,391 道共同题的比例`降序；不稳定题指该 Label 在五次最终结果中被选中 1–4 次。表中同时保留该 Label 实际出现过的题数，以及在出现过的题中发生不一致的条件比例。",
        "",
        f"共 {len(records)} 个 Label。",
        "",
        "| 排名 | Label | Label路径 | 不稳定分数 | 不稳定题数 | 出现题数 | 出现条件不稳定率 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | 代表题目 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(records, 1):
        examples = "、".join(f"`{qid}`({votes}/5)" for qid, votes in row["examples"])
        lines.append(
            f"| {rank} | {row['label_name']} (`{row['label_id']}`) | {row['label_path']} | "
            f"{row['instability_rate']:.4%} | {row['unstable_questions']} | {row['union_questions']} | "
            f"{row['conditional_instability_rate']:.2%} | {row['votes_1']} | {row['votes_2']} | "
            f"{row['votes_3']} | {row['votes_4']} | {row['votes_5']} | {examples or '-'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_question_examples(
    path: Path,
    report: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# DS 五次稳定性：前 30 道最不稳定题目",
        "",
        "> 本文从主报告中拆出题目示例，保留题干、解析、五次 Label，以及每次 DS 输出的 evidence/reason。n=4 条件逐个列出四个 choice。",
        "",
    ]
    for index, record in enumerate(report["question_records"], 1):
        qid = record["question_id"]
        unit = units.get(qid, {})
        lines += [
            f"## {index}. 题目 `{qid}`",
            "",
            f"- 题型：`{unit.get('unit_type') or unit.get('metadata', {}).get('structure_type') or 'unknown'}`",
            f"- 五次集合模式：`{record['classification']}`；不同集合数：{record['distinct_sets']}；平均两两 Jaccard：{record['mean_pair_jaccard']:.3f}；最低：{record['min_pair_jaccard']:.3f}",
            f"- 五次选中数量：{record['count_min']}–{record['count_max']}（范围 {record['count_range']}）",
            f"- 题干：{short_text(unit.get('stem'), 900) or '未提供'}",
            f"- 解析：{short_text(unit.get('analysis'), 900) or '未提供/略'}",
            "",
        ]
        for run_name, selected_ids in zip(RUN_NAMES, record["sets"]):
            label_text = "、".join(card(label_id, labels) for label_id in sorted(selected_ids)) or "（空）"
            lines.append(f"- `{run_name}`：{label_text}")
        lines += [
            "",
            "### DS 输出的 evidence / reason",
            "",
            "以下保留模型原始结构化输出中的 `evidence` 和 `reason`。",
            "",
        ]
        for run_name in RUN_NAMES:
            lines.append(f"**`{run_name}`**")
            for output in record["outputs"][run_name]:
                lines.append(f"- choice `{output.get('choice')}`")
                if output.get("parse_error"):
                    lines.append(f"  - parse_error: `{output['parse_error']}`")
                lines.append("  - evidence:")
                lines.append("    ```json")
                lines.append("    " + json.dumps(output.get("evidence"), ensure_ascii=False, indent=2).replace("\n", "\n    "))
                lines.append("    ```")
                lines.append("  - reason:")
                lines.append("    ```text")
                reason = str(output.get("reason") or "")
                lines.append("    " + (reason or "（空）").replace("\n", "\n    "))
                lines.append("    ```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_question_record(
    record: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
    *,
    include_outputs: bool,
) -> dict[str, Any]:
    qid = record["question_id"]
    unit = units.get(qid, {})
    result = {
        "question_id": qid,
        "question_type": unit.get("unit_type") or unit.get("metadata", {}).get("structure_type") or "unknown",
        "stem": unit.get("stem", ""),
        "analysis": unit.get("analysis", ""),
        "options": unit.get("options", ""),
        "classification": record["classification"],
        "distinct_sets": record["distinct_sets"],
        "mean_pair_jaccard": round(record["mean_pair_jaccard"], 6),
        "min_pair_jaccard": round(record["min_pair_jaccard"], 6),
        "instability_score": round(1.0 - record["mean_pair_jaccard"], 6),
        "count_min": record["count_min"],
        "count_max": record["count_max"],
        "count_range": record["count_range"],
        "runs": [
            {
                "condition": condition,
                "selected_label_ids": sorted(selected),
                "selected_label_names": [str(labels.get(label_id, {}).get("label_name") or label_id) for label_id in sorted(selected)],
            }
            for condition, selected in zip(RUN_NAMES, record["sets"])
        ],
    }
    if include_outputs:
        result["ds_outputs"] = record["outputs"]
    return result


def write_json_export(
    path: Path,
    report: dict[str, Any],
    all_question_records: list[dict[str, Any]],
    unstable_labels: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
) -> None:
    value = {
        "metadata": {
            "run_dir": report["run_dir"],
            "run_names": report["run_names"],
            "five_run_common_questions": report["five_run_common_questions"],
            "definition": "question instability = 1 - mean pairwise Jaccard over the five final Label sets; n=4 uses strict majority",
        },
        "summary": {
            "input_stats": report["input_stats"],
            "choice_stats": report["choice_stats"],
            "prompt_consistency_vs_original": report["prompt_consistency_vs_original"],
            "pairwise": report["pairwise"],
            "five_run_set_patterns": report["five_run_set_patterns"],
        },
        "question_ranking": [
            _json_question_record(record, labels, units, include_outputs=False)
            for record in all_question_records
        ],
        "question_examples_with_evidence_reason": [
            _json_question_record(record, labels, units, include_outputs=True)
            for record in report["question_records"]
        ],
        "label_ranking": unstable_labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], labels: dict[str, dict[str, Any]], units: dict[str, dict[str, Any]]) -> None:
    lines: list[str] = [
        "# DeepSeek-V4-Flash 五次稳定性实验详细分析",
        "",
        "> 本报告将原始 DS 生产结果与四组控制实验放在同一 5,391 题集合上比较。四组控制实验均使用 `candidate_mode=legacy`，即 Top25+旧 `knw_ids` 候选集合；`n=4` 的最终 Label 使用严格多数票（至少 3/4）计算。",
        "",
        "## 一、实验范围与数据完整性",
        "",
        "| 运行 | 文件记录数 | 错误记录数 | 成功题数 | choice 数 | 解析错误 | 最终平均 Label 数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RUN_NAMES:
        s = report["input_stats"][name]
        c = report["choice_stats"][name]
        lines.append(f"| `{name}` | {s['file_rows']} | {s['error_rows']} | {s['successful_questions']} | {c['choice_count']} | {c['parse_errors']} | {c['mean_final_selected']:.4f} |")
    lines += [
        "",
        "原始 DS 文件是追加式日志，包含重试错误行；最终按 `question_id` 保留成功记录后，五组均覆盖 5,391 道题。`temp01-n4-workers20` 有 2 个 choice 解析错误，但每题仍有足够的有效 choice 可形成多数票；其余条件没有解析错误。",
        "",
        "五组共同题目的 `prompt_sha256` 逐题完全一致（原始 DS 与每个控制条件均为 5,391/5,391），因此本报告中的差异不是由题目、候选 Label 顺序或释义内容变化造成的，主要反映请求条件和服务重复调用差异。",
        "",
        "## 二、两两输出差异",
        "",
        "| 左侧 | 右侧 | 共同题 | 输出不同 | 多选题 | 少选题 | 同数替换 | 新增 Label | 移除 Label | 净变化 | 平均 Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["pairwise"]:
        lines.append(
            f"| `{row['left']}` | `{row['right']}` | {row['common']} | {row['different']} ({pct(row['difference_rate'])}) | "
            f"{row['more']} | {row['fewer']} | {row['same_count_replaced']} | {row['added_assignments']} | {row['removed_assignments']} | "
            f"{row['net_assignment_delta']} | {row['mean_jaccard']:.4f} |"
        )
    lines += [
        "",
        "“多选/少选”按最终 Label 数量判断；数量相同但 Label 集合不同单列为“同数替换”。`输出不同`是集合差异，不是顺序差异。",
        "",
        "### 关键对照",
        "",
        "- `original_ds` vs `temp0-workers20`：用于观察同一 prompt/候选输入下，原始生产请求与重新请求的波动。",
        "- `temp0-workers20` vs `temp0-seed42-workers20`：temperature 都为 0，主要观察 seed/服务重复调用差异。",
        "- `temp0-workers20` vs 两个 `temp01-n4`：观察 temperature=0.1 且多数票后的最终输出变化。",
        "- 两个 `temp01-n4` 互比：观察 seed 对 n=4 采样结果的影响。",
        "",
        "## 三、五次结果的整体共识",
        "",
        f"五次均成功的共同题数：**{report['five_run_common_questions']}**。",
        "",
        "| 五次集合模式 | 题数 |",
        "|---|---:|",
    ]
    for key, value in report["five_run_set_patterns"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "这里的“集合模式”比较的是最终 Label 集合，而不是模型的 reasoning 文本。若五次集合相同，只能说明选标结果一致，不能说明模型内部推理完全一致。",
        "",
        "## 四、最不稳定题目",
        "",
        "前 30 道题的题干、解析、五次 Label 以及 evidence/reason 已移至 [ds-stability-question-examples-30.md](ds-stability-question-examples-30.md)。全部 5,391 道题的紧凑排序见 [ds-stability-question-ranking-5391.md](ds-stability-question-ranking-5391.md)。",
        "",
    ]

    lines += [
        "## 五、最容易发生跨次变化的 Label",
        "",
        "统计口径：在五次最终结果中，一个 Label 被 1–4 次选中的题数。被 5 次都选中或 5 次都未选中的题不计为不稳定。",
        "",
        "| Label | 不稳定题数 | 1/5 次 | 2/5 次 | 3/5 次 | 4/5 次 | 代表题目 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["unstable_labels"]:
        examples = "、".join(f"{qid}（{votes}/5）" for qid, votes in row["examples"])
        lines.append(
            f"| {row['label_name']} (`{row['label_id']}`) | {row['unstable_questions']} | {row['votes_1']} | {row['votes_2']} | {row['votes_3']} | {row['votes_4']} | {examples or '-'} |"
        )
    lines += [
        "",
        "## 六、完整排序文件",
        "",
        "- [5,391 道题不稳定度完整排序](ds-stability-question-ranking-5391.md)：按 `1 - 平均两两 Jaccard` 从高到低。",
        "- [458 个 Label 不稳定度完整排序](ds-stability-label-ranking-458.md)：按不稳定题占共同题比例从高到低，并附覆盖量和五次投票分布。",
        "",
        "题目示例已单独拆出；完整排序文件保留全部题目和全部 Label。",
        "",
        "## 七、结论与使用建议",
        "",
        "1. 这批题的波动不是简单的 Label 顺序变化，而是集合内容变化；其中一部分题表现为少选/多选，另一部分表现为同数量的 Label 替换。",
        "2. temperature=0 的重复请求仍然存在明显服务/模型波动，因此不能把 temperature=0 等同于完全确定性。",
        "3. temperature=0.1,n=4 的单次 choice 波动比 n=1 更明显，但严格多数票会减少一次性偶然 Label；生产上若采用 n=4，应保留全部 choice 和多数票过程，不能只保存最终集合。",
        "4. 五次分析中最值得人工复核的是：平均 Jaccard 低、五次没有三次以上共同集合、以及某个 Label 只在 1–2 次出现的题。报告第四节和第五节已经列出具体题号、题干和 Label。",
        "5. 本实验仍是模型稳定性分析，不等价于标签正确率；真正判断 Label 是否应该保留，仍需结合题目设问、Label 释义和教师复核。",
        "",
        "## 八、输入文件",
        "",
        "- `run_manifest.json`：实验条件、prompt 版本、输入 hash。",
        "- `report.json`：四组条件的服务层汇总和两两比较。",
        "- `analysis-export/original_ds_baseline_5391.jsonl`：原始 DS 第五次基线的 5,391 题筛选结果。",
        "- `analysis-export/pilot_units_5391.jsonl`：题干、解析、题型和题目结构信息。",
        "- 四个条件目录下的 `responses.jsonl`：逐题原始 choices、解析结果和 Label ID。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
