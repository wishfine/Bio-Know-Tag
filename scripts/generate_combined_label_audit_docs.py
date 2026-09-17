#!/usr/bin/env python3
"""Generate positive/negative combined Label audit Markdown with question evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def clip(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def overlap_prone(name: str) -> bool:
    return any(
        token in name
        for token in (
            "综合",
            "区别与联系",
            "对比",
            "关系",
            "概述",
            "文字信息类",
            "图形图像类",
            "数据图像类",
        )
    )


def positive_risk_level(
    metric: dict[str, Any], strategy: dict[str, Any] | None
) -> str:
    """Keep the 135-label positive-audit scope consistent across reports."""
    planned = int(metric["planned"])
    match_rate = float(metric["match_rate"])
    zero_rate = float(metric["zero_rate"])
    final_strategy = (strategy or {}).get("final_strategy") or {}
    status = final_strategy.get("status")
    manual_followup = bool(final_strategy.get("manual_followup_required"))
    if status == "taxonomy_hold":
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
    if manual_followup:
        return "P2_边界观察"
    return "S_正样本稳定"


def positive_score_bands(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    """Return mutually exclusive score bands used in the positive audit."""
    scores = [float(row["relevance_score"]) for row in rows]
    total = len(scores)
    irrelevant = sum(score < 0.10 for score in scores)
    related_below_definition = sum(0.10 <= score < 0.70 for score in scores)
    matched = sum(score >= 0.70 for score in scores)
    high_confidence = sum(score >= 0.80 for score in scores)
    if irrelevant + related_below_definition + matched != total:
        raise ValueError("positive score bands do not sum to the Label total")
    return {
        "total": total,
        "irrelevant": irrelevant,
        "irrelevant_rate": irrelevant / total if total else 0.0,
        "related_below_definition": related_below_definition,
        "related_below_definition_rate": (
            related_below_definition / total if total else 0.0
        ),
        "matched": matched,
        "matched_rate": matched / total if total else 0.0,
        "high_confidence": high_confidence,
        "high_confidence_rate": high_confidence / total if total else 0.0,
    }


def interpretation(
    label: dict[str, Any], positive: dict[str, Any], negative: dict[str, Any] | None
) -> tuple[str, str]:
    name = str(label["label_name"])
    p_count = int(positive["planned"])
    p_rate = float(positive["match_rate"])
    if negative is None:
        return (
            "负样本缺失",
            "无可用同父级兄弟硬负样本；不能判定释义是否偏宽。",
        )
    n_count = int(negative["hard_negative_total"])
    n_rate = float(negative["false_accept_rate"])
    if p_count < 300:
        return (
            "长尾待人工",
            "正样本少于300；比例不稳定，不根据正负指标自动改释义。",
        )
    if n_count < 20:
        return (
            "负样本不足",
            "硬负样本少于20，只作个案证据，不用百分比分档。",
        )
    if overlap_prone(name) and n_rate > 0.15:
        if p_rate < 0.55:
            return (
                "层级口径冲突",
                "该综合/比较/上位Label的兄弟题可能合理共标；正样本又显示旧体系将其当作兜底，应先明确层级规则。",
            )
        return (
            "天然重叠/伪负例",
            "该综合/比较/上位Label本就可覆盖部分兄弟原子题；高接受率不能直接证明释义偏宽。",
        )
    if p_rate >= 0.70 and n_rate <= 0.10:
        return (
            "正负边界稳定",
            "历史正样本覆盖较高，对兄弟题的接受率低，可作为当前稳定候选。",
        )
    if p_rate < 0.55 and n_rate <= 0.10:
        return (
            "旧标噪声/释义偏窄待分",
            "排除兄弟题的能力较好，但历史正样本覆盖低；先逐题判断旧误标还是释义漏项。",
        )
    if p_rate >= 0.70 and n_rate > 0.15:
        return (
            "偏宽或兄弟重叠",
            "正样本覆盖高，但大量兄弟题也被接受；需核对具体source→target是合理共标还是边界过宽。",
        )
    if p_rate < 0.55 and n_rate > 0.15:
        return (
            "边界冲突",
            "对历史正题覆盖低，同时又接受较多兄弟题；应优先复核名称、释义和历史ID口径。",
        )
    return (
        "轻度边界复核",
        "正覆盖或兄弟排除有一项未达稳定门槛，结合具体混淆对抽查。",
    )


def choose_positive_examples(
    label_id: str,
    results_by_label: dict[str, list[dict[str, Any]]],
    tasks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = sorted(
        results_by_label.get(label_id, []),
        key=lambda row: (float(row["relevance_score"]), str(row["question_id"])),
    )
    chosen: list[dict[str, Any]] = []
    if rows:
        chosen.append(rows[0])
    high = [row for row in rows if float(row["relevance_score"]) >= 0.80]
    if high:
        chosen.append(high[-1])
    output = []
    seen = set()
    for row in chosen:
        if row["task_id"] in seen:
            continue
        seen.add(row["task_id"])
        output.append({**row, "task": tasks[row["task_id"]]})
    return output


def choose_negative_examples(
    label_id: str,
    samples: dict[str, dict[str, Any]],
    negative_results_by_label: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = sorted(
        negative_results_by_label.get(label_id, []),
        key=lambda row: (-float(row["relevance_score"]), str(row["question_id"])),
    )
    accepted = [row for row in rows if float(row["relevance_score"]) >= 0.70]
    rejected = [row for row in reversed(rows) if float(row["relevance_score"]) < 0.70]
    chosen = accepted[:2] + rejected[:1]
    return [{**row, "sample": samples[row["task_id"]]} for row in chosen]


def generate(args: argparse.Namespace) -> tuple[Path, Path]:
    labels = {str(row["label_id"]): row for row in read_jsonl(args.labels)}
    strategies = {str(row["label_id"]): row for row in read_jsonl(args.strategies)}
    positive = {str(row["label_id"]): row for row in read_jsonl(args.positive_per_label)}
    negative = {str(row["label_id"]): row for row in read_jsonl(args.negative_per_label)}
    combined = {str(row["label_id"]): row for row in read_jsonl(args.combined)}
    corrected = {
        str(row["label_id"]): row for row in read_jsonl(args.corrected_assessments)
    }
    tasks = {str(row["pair_id"]): row for row in read_jsonl(args.positive_tasks)}
    positive_results_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.positive_results):
        positive_results_by_label[str(row["label_id"])].append(row)
    samples = {str(row["pair_id"]): row for row in read_jsonl(args.negative_samples)}
    negative_results_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.negative_results):
        negative_results_by_label[str(row["label_id"])].append(row)
    confusion_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.confusion_pairs):
        confusion_by_target[str(row["target_label_id"])].append(row)

    if (
        set(labels) != set(positive)
        or set(labels) != set(combined)
        or set(labels) != set(corrected)
    ):
        raise ValueError(
            "Label, positive, combined, and corrected inputs must cover the same 458 IDs"
        )
    if set(samples) != {
        str(row["task_id"])
        for rows in negative_results_by_label.values()
        for row in rows
    }:
        raise ValueError("hard-negative samples/results task IDs differ")

    items = []
    for label_id, label in labels.items():
        neg = negative.get(label_id)
        category, conclusion = interpretation(label, positive[label_id], neg)
        pairs = sorted(
            confusion_by_target.get(label_id, []),
            key=lambda row: (-float(row["false_accept_rate"]), -int(row["total"])),
        )
        items.append(
            {
                "label": label,
                "strategy": strategies.get(label_id, {}),
                "positive": positive[label_id],
                "negative": neg,
                "combined": combined[label_id],
                "corrected": corrected[label_id],
                "category": category,
                "conclusion": conclusion,
                "pairs": pairs,
                "positive_examples": choose_positive_examples(
                    label_id, positive_results_by_label, tasks
                ),
                "negative_examples": choose_negative_examples(
                    label_id, samples, negative_results_by_label
                ),
            }
        )

    category_order = {
        "边界冲突": 0,
        "旧标噪声/释义偏窄待分": 1,
        "偏宽或兄弟重叠": 2,
        "层级口径冲突": 3,
        "负样本缺失": 4,
        "负样本不足": 5,
        "长尾待人工": 6,
        "轻度边界复核": 7,
        "天然重叠/伪负例": 8,
        "正负边界稳定": 9,
    }
    items.sort(
        key=lambda item: (
            category_order[item["category"]],
            item["positive"]["match_rate"],
            -(item["negative"] or {}).get("false_accept_rate", -1),
            item["label"]["label_name"],
        )
    )
    category_counts = Counter(item["category"] for item in items)
    screen_counts = Counter(item["combined"]["final_screen"] for item in items)

    positive_problem_risks = {
        "P0_图谱冲突",
        "P0_明显异常",
        "P1_重点核验",
        "L0_极端长尾",
        "L1_长尾异常",
    }
    positive_problem_items = []
    for item in items:
        label_id = str(item["label"]["label_id"])
        risk = positive_risk_level(item["positive"], item["strategy"])
        manual_followup = bool(
            ((item["strategy"].get("final_strategy") or {}).get(
                "manual_followup_required"
            ))
        )
        if risk not in positive_problem_risks and not manual_followup:
            continue
        positive_problem_items.append(
            {
                **item,
                "positive_risk": risk,
                "score_bands": positive_score_bands(
                    positive_results_by_label[label_id]
                ),
            }
        )
    positive_problem_items.sort(
        key=lambda item: (
            -float(item["score_bands"]["irrelevant_rate"]),
            -float(item["score_bands"]["related_below_definition_rate"]),
            str(item["label"]["label_name"]),
        )
    )
    positive_problem_totals = Counter()
    for item in positive_problem_items:
        bands = item["score_bands"]
        for key in (
            "total",
            "irrelevant",
            "related_below_definition",
            "matched",
            "high_confidence",
        ):
            positive_problem_totals[key] += int(bands[key])

    positive_problem_ids = {
        str(item["label"]["label_id"]) for item in positive_problem_items
    }
    corrected_screen_counts = Counter(
        item["corrected"]["final_screen"] for item in items
    )
    corrected_totals = Counter()
    for item in items:
        row = item["corrected"]
        for key in (
            "hard_negative_total",
            "first_stage_accepted",
            "first_stage_rejected",
            "invalid_sibling_negatives",
            "unresolved",
            "valid_negative_count",
            "true_boundary_errors",
        ):
            corrected_totals[key] += int(row[key])
    corrected_totals["reasonable_colabel"] = sum(
        int((item["corrected"].get("colabel_decision_counts") or {}).get("合理共标", 0))
        for item in items
    )
    corrected_totals["source_insufficient"] = sum(
        int(
            (item["corrected"].get("colabel_decision_counts") or {}).get(
                "来源Label不足以描述该题", 0
            )
        )
        for item in items
    )
    corrected_totals["neither_sufficient"] = sum(
        int(
            (item["corrected"].get("colabel_decision_counts") or {}).get(
                "两侧Label均不充分", 0
            )
        )
        for item in items
    )
    additional_boundary_items = [
        item
        for item in items
        if str(item["label"]["label_id"]) not in positive_problem_ids
        and int(item["corrected"]["positive_count"]) >= 300
        and int(item["corrected"]["valid_negative_count"]) >= 20
        and item["corrected"].get("corrected_boundary_error_rate") is not None
        and float(item["corrected"]["corrected_boundary_error_rate"]) > 0.15
    ]
    additional_boundary_items.sort(
        key=lambda item: (
            -float(item["corrected"]["corrected_boundary_error_rate"]),
            str(item["label"]["label_name"]),
        )
    )

    full = [
        "# 高中生物458个Label正负样本联合复核",
        "",
        "## 实验口径",
        "",
        "- 正样本：179,568个独立题—历史Label对，检验当前释义对历史已打题的覆盖。",
        "- 硬负样本：21,647个“高置信兄弟Label正题→目标Label”对，检验兄弟边界排除能力。",
        "- 硬负样本不是金标负例：兄弟Label可能合理共标，尤其是综合、比较、上位和信息形式Label。",
        "- 第一阶段只测“兄弟题接受率”；对接受的5,555条再做V2匿名候选最小充分集合二次Judge。",
        "- V2仍是DS Judge证据，且“最小充分集合”比业务可接受的合理多标更严。因此校正指标称为“目标Label排除率”，不直接称为人工金标错标率。",
        "",
        "## 总体结果",
        "",
        "- 正样本match：68.53%。",
        f"- 第一阶段兄弟题接受：{corrected_totals['first_stage_accepted']:,}/{corrected_totals['hard_negative_total']:,}（{corrected_totals['first_stage_accepted'] / corrected_totals['hard_negative_total']:.2%}）。",
        f"- V2二次裁决：合理共标{corrected_totals['reasonable_colabel']:,}；来源Label不足{corrected_totals['source_insufficient']:,}；两侧都不充分{corrected_totals['neither_sufficient']:,}；目标Label被最小集合排除{corrected_totals['true_boundary_errors']:,}；无法判断{corrected_totals['unresolved']:,}。",
        f"- 移除{corrected_totals['invalid_sibling_negatives']:,}条伪负例后，有效负样本{corrected_totals['valid_negative_count']:,}，目标Label排除{corrected_totals['true_boundary_errors']:,}，校正排除率{corrected_totals['true_boundary_errors'] / corrected_totals['valid_negative_count']:.2%}。",
        "",
        "## 正样本分档策略",
        "",
        "正样本分档只回答“当前老师释义能否覆盖历史已打Label的独立题”，不能判定释义是否偏宽。规则按以下顺序命中：",
        "",
        "| 分档 | 规则 | 含义 |",
        "|---|---|---|",
        "| P0_图谱冲突 | 既有策略状态为`taxonomy_hold` | Label名与释义/taxonomy冲突，暂停自动定标 |",
        "| L0_极端长尾 | 正样本<30 | 比例不稳定，应逐题看 |",
        "| L1_长尾异常 | 30–299题，且match<40%或0分率≥50% | 长尾中的明显异常 |",
        "| L2_长尾待核 | 30–299题，且未命中L1 | 证据不足，不自动改释义 |",
        "| P0_明显异常 | 正样本≥300，且match<20%或0分率≥60% | 优先核查旧ID误挂/语义漂移 |",
        "| P1_重点核验 | 正样本≥300，且match<55%或0分率≥30% | 区分旧标错与释义漏项 |",
        "| P2_边界观察 | match<70%、初筛非稳定，或既有策略要求人工跟进 | 中等覆盖，需结合负样本 |",
        "| S_正样本稳定 | 未命中以上任何规则 | 只表示正样本覆盖稳定 |",
        "",
        "旧135个问题Label集合由`P0_图谱冲突/P0_明显异常/P1_重点核验/L0_极端长尾/L1_长尾异常`加上既有策略中明确要求人工跟进的Label组成。",
        "",
        "## 135个问题Label的正样本分数分段",
        "",
        "- 本表仅收录正样本阶段筛出的135个问题Label，与《高中生物需重点关注的Label与释义证据》口径一致。",
        "- 三档互斥：`<0.10`为基本无关；`0.10–0.69`为与Label相关、但未达到当前释义的主要考查要求；`≥0.70`为匹配。",
        "- `相关但未达到释义要求`不等于错标；它主要用来定位上位/综合Label口径、历史弱标与释义边界之间的冲突。",
        f"- 合计：{positive_problem_totals['total']:,}题；基本无关{positive_problem_totals['irrelevant']:,}题（{positive_problem_totals['irrelevant'] / positive_problem_totals['total']:.2%}）；相关但未达到释义要求{positive_problem_totals['related_below_definition']:,}题（{positive_problem_totals['related_below_definition'] / positive_problem_totals['total']:.2%}）；已匹配{positive_problem_totals['matched']:,}题（{positive_problem_totals['matched'] / positive_problem_totals['total']:.2%}）。",
        "",
        "| Label | ID | 风险层 | 总题数 | 基本无关题/总题数 | 基本无关占比 | 相关但未达到释义要求/总题数 | 边界题占比 | 匹配题/总题数 | 匹配占比 | 高置信题(≥0.80)/总题数 | 高置信占比 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in positive_problem_items:
        label = item["label"]
        bands = item["score_bands"]
        total = int(bands["total"])
        full.append(
            f"| {md(label['label_name'])} | `{label['label_id']}` | {item['positive_risk']} | {total} | "
            f"{bands['irrelevant']}/{total} | {float(bands['irrelevant_rate']):.2%} | "
            f"{bands['related_below_definition']}/{total} | {float(bands['related_below_definition_rate']):.2%} | "
            f"{bands['matched']}/{total} | {float(bands['matched_rate']):.2%} | "
            f"{bands['high_confidence']}/{total} | {float(bands['high_confidence_rate']):.2%} |"
        )
    full.extend(
        [
            "",
            "## V2校正后联合分档",
            "",
            "| 联合分档 | Label数 | 规则与含义 |",
            "|---|---:|---|",
        ]
    )
    corrected_meanings = {
        "A_STABLE_CANDIDATE": "正样本≥300、match≥70%、有效负样本≥20、校正排除率≤10%；当前稳定候选。",
        "B_MINOR_BOUNDARY_REVIEW": "未命中A/C/D/E的中间档；这是异质兜底组，不能统一解读为轻度问题。",
        "C_NARROW_OR_LEGACY_NOISE_REVIEW": "正样本match<55%且校正排除率≤10%；优先分旧标噪声与释义偏窄。",
        "D_BROAD_BOUNDARY_REVIEW": "正样本match≥70%且校正排除率>15%；在最小集合口径下存在边界风险。",
        "E_BOUNDARY_CONFLICT_REVIEW": "正样本match<55%且校正排除率>15%；正负两侧均冲突。",
        "U_INSUFFICIENT_VALID_NEGATIVES": "正样本≥300，但有效负样本<20；不按比例自动下结论。",
        "U_LONG_TAIL_REVIEW": "正样本<300；长尾优先，不按比例自动改释义。",
    }
    for screen in (
        "A_STABLE_CANDIDATE",
        "B_MINOR_BOUNDARY_REVIEW",
        "C_NARROW_OR_LEGACY_NOISE_REVIEW",
        "D_BROAD_BOUNDARY_REVIEW",
        "E_BOUNDARY_CONFLICT_REVIEW",
        "U_INSUFFICIENT_VALID_NEGATIVES",
        "U_LONG_TAIL_REVIEW",
    ):
        full.append(
            f"| {screen} | {corrected_screen_counts[screen]} | {corrected_meanings[screen]} |"
        )
    full.extend(
        [
            "",
            "## 77个新增边界风险候选（旧135之外）",
            "",
            "这77个Label的选入理由是一致的：",
            "",
            "1. 不在正样本旧135问题集合中；",
            "2. 正样本不少于300题，避免长尾比例波动；",
            "3. 移除合理共标、来源Label不足、两侧都不充分后，有效负样本不少于20题；",
            "4. V2匿名最小充分集合中，目标Label被排除的校正比例>15%。",
            "",
            f"按当前结果共{len(additional_boundary_items)}个：58个`D_BROAD_BOUNDARY_REVIEW` + 19个落入兜底B档但校正排除率仍>15%的Label。",
            "",
            "这些Label的证据只说明“在当前DS最小集合策略下经常被排除”。如果原source Label有误，或业务允许合理冗余多标，就不能直接归因为老师释义偏宽。",
            "",
            "| Label | ID | 正样本分档 | 正题n | 正match | 有效负n | 目标排除 | 校正排除率 | 联合分档 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in additional_boundary_items:
        label, pos, row = item["label"], item["positive"], item["corrected"]
        full.append(
            f"| {md(label['label_name'])} | `{label['label_id']}` | "
            f"{positive_risk_level(pos, item['strategy'])} | {pos['planned']} | "
            f"{pct(pos['match_rate'])} | {row['valid_negative_count']} | "
            f"{row['true_boundary_errors']} | {pct(row['corrected_boundary_error_rate'])} | "
            f"{row['final_screen']} |"
        )
    full.extend(
        [
            "",
            "## 458个Label校正后联合分析",
            "",
            "| Label | ID | 正样本分档 | 正题n | 正match | 原始负n | 第一阶段接受 | 伪负例移除 | 有效负n | 目标排除 | 校正排除率 | 联合分档 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in items:
        label, pos, row = item["label"], item["positive"], item["corrected"]
        full.append(
            f"| {md(label['label_name'])} | `{label['label_id']}` | "
            f"{positive_risk_level(pos, item['strategy'])} | {pos['planned']} | "
            f"{pct(pos['match_rate'])} | {row['hard_negative_total']} | "
            f"{row['first_stage_accepted']} | {row['invalid_sibling_negatives']} | "
            f"{row['valid_negative_count']} | {row['true_boundary_errors']} | "
            f"{pct(row['corrected_boundary_error_rate'])} | {row['final_screen']} |"
        )
    full.extend(
        [
            "",
            "## 结论边界与后续验证",
            "",
            "1. V2已经显式识别1,445条“来源Label不足”和24条“两侧都不充分”，证明原source Label不能当金标。",
            "2. V2 Prompt要求最小充分集合，会排斥部分业务可接受的合理上下位/冗余多标；因此2,683条是风险证据，不是人工真值。",
            "3. 如要估计“只要合理即可保留”的真实错标率，需用V3对source/target逐个独立T/F，禁止因另一Label更具体就拒绝当前Label。",
            "4. 当前77个新增候选应给老师看代表题，判断是原source错、合理共标、最小集合偏好，还是target确实不应打。",
            "",
        ]
    )

    focus_categories = {
        "边界冲突",
        "旧标噪声/释义偏窄待分",
        "偏宽或兄弟重叠",
        "层级口径冲突",
        "负样本缺失",
        "负样本不足",
    }
    focus_items = [
        item
        for item in items
        if item["category"] in focus_categories
        or (
            item["category"] == "长尾待人工"
            and (
                int(item["positive"]["planned"]) < 30
                or float((item["negative"] or {}).get("false_accept_rate", 0)) > 0.30
            )
        )
        or bool(
            ((item["strategy"].get("final_strategy") or {}).get("manual_followup_required"))
        )
    ]
    focus = [
        "# 高中生物Label边界重点复核（正负题证据）",
        "",
        f"本文收录{len(focus_items)}个需要学科重点确认的Label。兄弟硬负题可能是合理共标，因此每个高接受样本都保留source→target关系和题目正文，不依赖百分比直接改释义。",
        "",
    ]
    for index, item in enumerate(focus_items, 1):
        label, pos, neg = item["label"], item["positive"], item["negative"]
        strategy = item["strategy"]
        focus.extend(
            [
                f"## {index}. {label['label_name']}（{item['category']}）",
                "",
                f"- Label ID：`{label['label_id']}`",
                f"- 路径：{label.get('label_path', '')}",
                f"- 正样本：n={pos['planned']}，match={pct(pos['match_rate'])}，≥0.80={pct(pos['high_match_rate'])}，0分={pct(pos['zero_rate'])}",
                f"- 兄弟硬负题：n={neg['hard_negative_total'] if neg else 0}，接受率={pct(neg['false_accept_rate']) if neg else '-'}",
                f"- 自动筛查：{item['combined']['final_screen']}",
                f"- 校正解读：{item['conclusion']}",
                f"- 此前复核：{(strategy.get('second_review') or {}).get('rationale', '-')}",
                "",
                "### 老师释义",
                "",
                f"- definition：{label.get('definition', '')}",
                f"- core_concepts：{label.get('core_concepts', '')}",
                f"- distinctions：{label.get('distinctions', '')}",
                "",
                "### source→target混淆对",
                "",
            ]
        )
        if item["pairs"]:
            focus.extend(["| source | target | 接受/总数 | 接受率 |", "|---|---|---:|---:|"])
            for pair in item["pairs"][:5]:
                focus.append(
                    f"| {md(pair['source_label_name'])} | {md(pair['target_label_name'])} | "
                    f"{pair['false_accept']}/{pair['total']} | {pct(pair['false_accept_rate'])} |"
                )
        else:
            focus.append("无可用兄弟硬负样本。")
        focus.extend(["", "### 正样本证据", ""])
        for example in item["positive_examples"]:
            task = example["task"]
            focus.extend(
                [
                    f"- `{example['question_id']}` score={float(example['relevance_score']):.2f}：{clip(task.get('stem'), 420)}",
                    f"  - 解析摘要：{clip(task.get('analysis'), 420) or '-'}",
                ]
            )
        focus.extend(["", "### 兄弟硬负题证据", ""])
        if item["negative_examples"]:
            for example in item["negative_examples"]:
                sample = example["sample"]
                focus.extend(
                    [
                        f"#### `{example['question_id']}` score={float(example['relevance_score']):.2f}",
                        "",
                        f"- source Label：{'、'.join(sample.get('source_label_names') or [])}",
                        f"- target Label：{label['label_name']}",
                        f"- 题干：{clip(sample.get('stem'), 520)}",
                        f"- 选项：{clip(sample.get('options'), 420) or '-'}",
                        f"- 答案：{clip(sample.get('answer_text'), 180) or '-'}",
                        f"- 解析摘要：{clip(sample.get('analysis'), 520) or '-'}",
                        "- 学科需判定：该source题是否也应合理共标target？若否，target释义的哪个边界需收窄？",
                        "",
                    ]
                )
        else:
            focus.extend(["无负样本证据。", ""])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "biology-458-label-positive-negative-combined-review.md"
    focus_path = args.output_dir / "biology-label-boundary-focus-positive-negative-evidence.md"
    full_path.write_text("\n".join(full), encoding="utf-8")
    focus_path.write_text("\n".join(focus), encoding="utf-8")
    return full_path, focus_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--positive-tasks", type=Path, required=True)
    parser.add_argument("--positive-results", type=Path, required=True)
    parser.add_argument("--positive-per-label", type=Path, required=True)
    parser.add_argument("--negative-samples", type=Path, required=True)
    parser.add_argument("--negative-results", type=Path, required=True)
    parser.add_argument("--negative-per-label", type=Path, required=True)
    parser.add_argument("--confusion-pairs", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--corrected-assessments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    full, focus = generate(args)
    print(json.dumps({"combined_review": str(full), "focus_review": str(focus)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
