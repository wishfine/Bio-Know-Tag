#!/usr/bin/env python3
"""Generate detailed Markdown reports for the Label-definition ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def load_by_key(path: str | Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        key = str(row.get("task_id") or row.get("pair_id") or "")
        if key:
            result[key] = row
    return result


def short(value: Any, limit: int = 420) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-questions", type=int, default=30)
    parser.add_argument("--top-labels", type=int, default=40)
    args = parser.parse_args()

    root = args.run_dir
    name_only = load_by_key(root / "ds-name-only/results.jsonl")
    with_definition = load_by_key(root / "ds-name-plus-definition/results.jsonl")
    tasks: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(root / "paired_tasks.jsonl"):
        key = str(row.get("pair_id") or row.get("task_id") or "")
        if not key:
            continue
        # Prefer the full-definition row because it carries the definition card.
        if key not in tasks or row.get("condition") == "name_plus_definition":
            tasks[key] = row
    labels = {
        str(row["label_id"]): row
        for row in read_jsonl(args.labels)
        if row.get("label_id")
    }
    common = sorted(set(name_only) & set(with_definition))

    statuses = Counter()
    deltas: list[float] = []
    changed_pairs: list[dict[str, Any]] = []
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "same_match": 0,
        "same_nonmatch": 0,
        "false_to_true": 0,
        "true_to_false": 0,
        "score_delta_sum": 0.0,
        "abs_score_delta_sum": 0.0,
        "positive_delta_sum": 0.0,
        "negative_delta_sum": 0.0,
        "examples": [],
    })
    by_source = Counter()
    by_stratum = Counter()
    for key in common:
        left = name_only[key]
        right = with_definition[key]
        task = tasks.get(key, {})
        label_id = str(left.get("label_id") or right.get("label_id") or task.get("label_id") or "")
        question_id = str(left.get("question_id") or right.get("question_id") or task.get("question_id") or "")
        left_match = bool(left.get("match"))
        right_match = bool(right.get("match"))
        left_score = float(left.get("relevance_score") or 0.0)
        right_score = float(right.get("relevance_score") or 0.0)
        delta = right_score - left_score
        if left_match and right_match:
            status = "same_match"
        elif not left_match and not right_match:
            status = "same_nonmatch"
        elif not left_match and right_match:
            status = "false_to_true"
        else:
            status = "true_to_false"
        statuses[status] += 1
        deltas.append(delta)
        source = str(task.get("sample_source") or "unknown")
        stratum = str(task.get("stratum") or "unknown")
        by_source[(source, status)] += 1
        by_stratum[(stratum, status)] += 1
        label_stats = by_label[label_id]
        label_stats["total"] += 1
        label_stats[status] += 1
        label_stats["score_delta_sum"] += delta
        label_stats["abs_score_delta_sum"] += abs(delta)
        if delta > 0:
            label_stats["positive_delta_sum"] += delta
        else:
            label_stats["negative_delta_sum"] += delta
        pair = {
            "pair_id": key,
            "question_id": question_id,
            "label_id": label_id,
            "label_name": str(task.get("label_name") or labels.get(label_id, {}).get("label_name") or ""),
            "label_path": str(task.get("label_path") or labels.get(label_id, {}).get("label_path") or ""),
            "match_rate": task.get("match_rate"),
            "stratum": stratum,
            "sample_source": source,
            "status": status,
            "name_only_match": left_match,
            "definition_match": right_match,
            "name_only_score": left_score,
            "definition_score": right_score,
            "score_delta": round(delta, 6),
            "abs_score_delta": round(abs(delta), 6),
            "stem": str(task.get("stem") or (task.get("question") or {}).get("stem") or ""),
            "options": str(task.get("options") or (task.get("question") or {}).get("options") or ""),
            "answer_text": str(task.get("answer_text") or (task.get("question") or {}).get("answer_text") or ""),
            "analysis": str(task.get("analysis") or (task.get("question") or {}).get("analysis") or ""),
            "image_context_missing": bool(task.get("image_context_missing")),
        }
        by_question[question_id].append(pair)
        if status in {"false_to_true", "true_to_false"} or abs(delta) >= 0.20:
            changed_pairs.append(pair)

    for label_id, stats in by_label.items():
        total = stats["total"]
        stats["label_id"] = label_id
        stats["label_name"] = str(labels.get(label_id, {}).get("label_name") or "")
        stats["label_path"] = str(labels.get(label_id, {}).get("label_path") or "")
        stats["status_changed"] = stats["false_to_true"] + stats["true_to_false"]
        stats["status_change_rate"] = stats["status_changed"] / total if total else 0.0
        stats["mean_score_delta"] = stats["score_delta_sum"] / total if total else 0.0
        stats["mean_abs_score_delta"] = stats["abs_score_delta_sum"] / total if total else 0.0
        stats["net_match_delta"] = stats["false_to_true"] - stats["true_to_false"]
        stats["examples"] = sorted(
            ({"question_id": p["question_id"], "status": p["status"], "score_delta": p["score_delta"]} for p in changed_pairs if p["label_id"] == label_id),
            key=lambda row: (-abs(row["score_delta"]), row["question_id"]),
        )[:5]

    # Include labels with no selected pair defensively, although this run has all 458.
    for label_id, label in labels.items():
        by_label.setdefault(label_id, {
            "label_id": label_id,
            "label_name": str(label.get("label_name") or ""),
            "label_path": str(label.get("label_path") or ""),
            "total": 0,
            "same_match": 0,
            "same_nonmatch": 0,
            "false_to_true": 0,
            "true_to_false": 0,
            "status_changed": 0,
            "status_change_rate": 0.0,
            "mean_score_delta": 0.0,
            "mean_abs_score_delta": 0.0,
            "net_match_delta": 0,
            "examples": [],
        })

    label_ranking = sorted(
        by_label.values(),
        key=lambda row: (-row.get("status_change_rate", 0), -row.get("status_changed", 0), -row.get("mean_abs_score_delta", 0), row.get("label_name", "")),
    )
    changed_pairs.sort(key=lambda row: (-row["abs_score_delta"], row["status"], row["question_id"], row["label_id"]))
    question_ranking = []
    for question_id, pairs in by_question.items():
        severe = [p for p in pairs if p["status"] in {"false_to_true", "true_to_false"} or p["abs_score_delta"] >= 0.20]
        if not severe:
            continue
        question_ranking.append({
            "question_id": question_id,
            "stem": severe[0]["stem"],
            "options": severe[0]["options"],
            "answer_text": severe[0]["answer_text"],
            "analysis": severe[0]["analysis"],
            "image_context_missing": severe[0]["image_context_missing"],
            "changed_pair_count": len(severe),
            "status_changed_count": sum(p["status"] in {"false_to_true", "true_to_false"} for p in severe),
            "max_abs_score_delta": max(p["abs_score_delta"] for p in severe),
            "sum_abs_score_delta": round(sum(p["abs_score_delta"] for p in severe), 6),
            "pairs": severe,
        })
    question_ranking.sort(key=lambda row: (-row["status_changed_count"], -row["max_abs_score_delta"], -row["sum_abs_score_delta"], row["question_id"]))

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_main_report(output, root, common, statuses, deltas, by_source, by_stratum, label_ranking, question_ranking, changed_pairs)
    write_question_report(output.with_name("definition-ablation-question-ranking.md"), question_ranking)
    write_label_report(output.with_name("definition-ablation-label-ranking.md"), label_ranking)
    with (output.parent / "definition-ablation-changed-pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in changed_pairs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "paired_results": len(common),
        "question_ranking": len(question_ranking),
        "changed_pairs": len(changed_pairs),
        "label_ranking": len(label_ranking),
        "change_counts": dict(statuses),
    }, ensure_ascii=False, indent=2))
    return 0


def write_main_report(path: Path, root: Path, common: list[str], statuses: Counter, deltas: list[float], by_source: Counter, by_stratum: Counter, label_ranking: list[dict[str, Any]], question_ranking: list[dict[str, Any]], changed_pairs: list[dict[str, Any]]) -> None:
    total = len(common)
    match_name = statuses["same_match"] + statuses["true_to_false"]
    match_definition = statuses["same_match"] + statuses["false_to_true"]
    lines = [
        "# 生物 Label 释义消融实验详细分析",
        "",
        "> 本实验比较同一批题目-Label 对在两种输入下的 DS 判断：A 为仅发送 Label 名称；B 为发送 Label 名称 + Label 释义卡片。历史 Label 只是弱监督，不是人工金标准。",
        "",
        "## 一、总体结果",
        "",
        f"- 配对题目-Label 数：**{total}**（458 个 Label，全部完成）",
        f"- 仅名称匹配：**{match_name}/{total} = {match_name / total:.2%}**",
        f"- 名称+释义匹配：**{match_definition}/{total} = {match_definition / total:.2%}**",
        f"- 匹配数量变化：**{match_definition - match_name:+d}**（{(match_definition - match_name) / total:+.2%}）",
        f"- 平均 relevance_score 变化：**{statistics.mean(deltas):+.4f}**",
        "",
        "| 变化类型 | 题目-Label 对 | 占比 | 含义 |",
        "|---|---:|---:|---|",
        f"| 保持匹配 | {statuses['same_match']} | {statuses['same_match'] / total:.2%} | 两种输入都判断为匹配 |",
        f"| 保持不匹配 | {statuses['same_nonmatch']} | {statuses['same_nonmatch'] / total:.2%} | 两种输入都判断为不匹配 |",
        f"| False → True | {statuses['false_to_true']} | {statuses['false_to_true'] / total:.2%} | 释义补足了名称不足的覆盖 |",
        f"| True → False | {statuses['true_to_false']} | {statuses['true_to_false'] / total:.2%} | 释义收紧了名称可能过宽的判断 |",
        f"| 判断状态发生变化 | {statuses['false_to_true'] + statuses['true_to_false']} | {(statuses['false_to_true'] + statuses['true_to_false']) / total:.2%} | False→True 或 True→False |",
        "",
        "## 二、按抽样来源和 Label 分层",
        "",
        "| 抽样来源/分层 | False→True | True→False | 状态变化合计 |",
        "|---|---:|---:|---:|",
    ]
    for source in sorted({key[0] for key in by_source}):
        lines.append(f"| `{source}` | {by_source[(source, 'false_to_true')]} | {by_source[(source, 'true_to_false')]} | {by_source[(source, 'false_to_true')] + by_source[(source, 'true_to_false')]} |")
    for stratum in sorted({key[0] for key in by_stratum}):
        lines.append(f"| `{stratum}` | {by_stratum[(stratum, 'false_to_true')]} | {by_stratum[(stratum, 'true_to_false')]} | {by_stratum[(stratum, 'false_to_true')] + by_stratum[(stratum, 'true_to_false')]} |")
    lines += [
        "",
        "## 三、变化最严重的 Label",
        "",
        "完整 458 个 Label 排名见 [definition-ablation-label-ranking.md](definition-ablation-label-ranking.md)。这里列出前 40 个；排序优先看状态变化率，再看状态变化数量和平均绝对分数变化。样本量为 5 或 10，不能把低样本 Label 的比例当作稳定结论。",
        "",
        "| 排名 | Label | 样本数 | 状态变化 | 变化率 | False→True | True→False | 平均分数变化 | 代表题 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(label_ranking[:40], 1):
        examples = "、".join(f"`{x['question_id']}`({x['status']})" for x in row["examples"][:3])
        lines.append(f"| {rank} | {row['label_name']} (`{row['label_id']}`) | {row['total']} | {row['status_changed']} | {row['status_change_rate']:.2%} | {row['false_to_true']} | {row['true_to_false']} | {row['mean_score_delta']:+.3f} | {examples or '-'} |")
    lines += [
        "",
        "## 四、变化最严重的题目",
        "",
        "完整题目排名见 [definition-ablation-question-ranking.md](definition-ablation-question-ranking.md)。下面列出前 30 道题；每条包含发生变化的 Label 以及两种条件的分数和状态。",
        "",
    ]
    for rank, row in enumerate(question_ranking[:30], 1):
        lines += [
            f"### {rank}. 题目 `{row['question_id']}`",
            "",
            f"- 状态变化 Label 数：{row['status_changed_count']}；严重变化对数：{row['changed_pair_count']}；最大绝对分数变化：{row['max_abs_score_delta']:.2f}",
            f"- 题干：{short(row['stem'], 900)}",
            f"- 选项：{short(row['options'], 700)}",
            f"- 答案：{short(row['answer_text'], 300)}",
            "- 变化 Label：",
        ]
        for pair in row["pairs"]:
            lines.append(f"  - `{pair['label_name']}`：`{pair['status']}`，{pair['name_only_score']:.2f} → {pair['definition_score']:.2f}；来源 `{pair['sample_source']}`，分层 `{pair['stratum']}`")
        lines.append("")
    lines += [
        "## 五、解释与限制",
        "",
        "1. 释义加入后总体匹配数几乎不变，但内部发生了约 20% 的状态重排：一部分题被释义补回，另一部分题被释义收紧。",
        "2. False→True 更像“Label 名称过短/名称不足”；True→False 更像“名称过宽、历史误挂或释义明确排除了该题”。单凭该实验不能区分这三种原因。",
        "3. 该实验结果文件只保存 `match` 和 `relevance_score`，没有 DS 的自然语言 `evidence/reason`；因此本报告可以定位严重题目，但不能直接解释 DS 为什么改变。",
        "4. 低匹配率 Label 抽 10 题，高匹配率 Label 抽 5 题；Label 之间样本量不均，需同时看变化率和变化数量。",
        "5. 这不是人工正确率实验；最终仍需要结合题干、解析、Label 释义和教师复核。",
        "",
        "## 六、输入与输出",
        "",
        f"- 运行目录：`{root}`",
        "- `ds-name-only/results.jsonl`：仅 Label 名称条件",
        "- `ds-name-plus-definition/results.jsonl`：名称+释义条件",
        "- `paired_tasks.jsonl`：题目、Label、释义和抽样信息",
        "- `definition-ablation-question-ranking.md`：全部严重变化题目排序",
        "- `definition-ablation-label-ranking.md`：全部 458 个 Label 排序",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_question_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 释义消融：题目严重变化完整排序",
        "",
        "> 排序优先级：状态变化 Label 数 → 最大绝对分数变化 → 该题所有变化对的绝对分数变化总和。每行展开一个题目及其严重变化 Label。",
        "",
        f"共 {len(rows)} 道出现严重变化的题目。",
        "",
        "| 排名 | 题目ID | 状态变化Label数 | 严重变化对数 | 最大绝对分数变化 | 题干 | 变化Label及方向 |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(rows, 1):
        changes = "；".join(f"{p['label_name']}({p['status']},{p['name_only_score']:.2f}→{p['definition_score']:.2f})" for p in row["pairs"])
        lines.append(f"| {rank} | `{row['question_id']}` | {row['status_changed_count']} | {row['changed_pair_count']} | {row['max_abs_score_delta']:.2f} | {short(row['stem'], 300)} | {changes} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 释义消融：458 个 Label 严重变化完整排序",
        "",
        "> 排序优先级：状态变化率 → 状态变化数量 → 平均绝对分数变化。请同时查看样本数，5 题 Label 的单个变化会造成 20 个百分点的变化率。",
        "",
        "| 排名 | Label | 路径 | 样本数 | 状态变化 | 变化率 | False→True | True→False | 平均分数变化 | 平均绝对变化 | 代表题 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, 1):
        examples = "、".join(f"`{x['question_id']}`({x['status']})" for x in row["examples"])
        lines.append(f"| {rank} | {row['label_name']} (`{row['label_id']}`) | {row['label_path']} | {row['total']} | {row['status_changed']} | {row['status_change_rate']:.2%} | {row['false_to_true']} | {row['true_to_false']} | {row['mean_score_delta']:+.3f} | {row['mean_abs_score_delta']:.3f} | {examples or '-'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
