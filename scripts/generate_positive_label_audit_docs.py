#!/usr/bin/env python3
"""Generate traceable Markdown audits from standalone positive-coverage results."""

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


def md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def clip(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def risk_level(metric: dict[str, Any], strategy: dict[str, Any] | None) -> str:
    planned = int(metric["planned"])
    match_rate = float(metric["match_rate"])
    zero_rate = float(metric["zero_rate"])
    status = ((strategy or {}).get("final_strategy") or {}).get("status")
    manual_followup = bool(
        ((strategy or {}).get("final_strategy") or {}).get("manual_followup_required")
    )
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


def cause_hypothesis(
    label: dict[str, Any], metric: dict[str, Any], strategy: dict[str, Any] | None
) -> str:
    name = str(label["label_name"])
    match_rate = float(metric["match_rate"])
    zero_rate = float(metric["zero_rate"])
    status = ((strategy or {}).get("final_strategy") or {}).get("status")
    manual_followup = bool(
        ((strategy or {}).get("final_strategy") or {}).get("manual_followup_required")
    )
    if status == "taxonomy_hold":
        return "Label名与释义已知冲突，不应由样本实验自动修复。"
    if any(token in name for token in ("综合", "协调配合", "区别与联系", "关系")) and match_rate < 0.70:
        return "历史题库可能把该综合/联动Label当作章节兜底，而当前释义要求多模块真正联动。"
    if any(token in name for token in ("实验", "探究", "观察", "制作", "调查", "模拟活动")) and match_rate < 0.70:
        return "历史标注可能混合了“实验操作/设计”与“相关原理/章节题”，需核对任务形态。"
    if any(token in name for token in ("应用", "成果", "药品")) and match_rate < 0.70:
        return "历史标注可能把上位工程原理、操作步骤与具体应用/成果混标。"
    if any(token in name for token in ("数据图像类", "图形图像类", "文字信息类")):
        return "这是信息形式/能力维度，与学科知识Label口径不同，应继续单独路由。"
    if zero_rate >= 0.60:
        return "大量历史题与当前释义完全无关，更像旧ID语义/映射漂移，不宜直接扩写释义迁就旧题。"
    if match_rate < 0.55 and zero_rate < 0.15:
        return "多数题与Label有关但未达到“主要考查”，需在上位标签口径与最小充分知识集之间做选择。"
    if match_rate < 0.55:
        return "低分中同时存在弱相关和完全无关题，需分开核对历史误标与释义漏项。"
    if int(metric["planned"]) < 300:
        return "独立题证据少于300，比例波动大，当前只能做长尾风险提示。"
    if manual_followup:
        return "正样本覆盖虽稳定，但此Label在既有释义审核中已记录边界风险，仍需结合负样本核验。"
    return "正样本覆盖基本稳定；是否偏宽仍要等待硬负样本结果。"


def recommended_action(
    risk: str, label: dict[str, Any], metric: dict[str, Any]
) -> str:
    name = str(label["label_name"])
    if risk == "P0_图谱冲突":
        return "暂停自动最终定标，请老师确认Label名或释义哪一侧需修正。"
    if risk == "L0_极端长尾":
        return "不据百分比改释义；所有样本逐题人工看，并优先补题。"
    if risk == "L1_长尾异常":
        return "全量人工复核该长尾Label的独立题，先清理旧误标，再判断是否补释义。"
    if risk.startswith("P0") or risk.startswith("P1"):
        if "综合" in name:
            return "保留当前释义，先剔除只考单一子模块的旧正样本；综合Label必须有联动证据。"
        return "先抽查本文列出的0分/弱相关题，判定“旧标错”或“释义漏”；不直接放宽。"
    if risk.startswith("L2"):
        return "释义暂不改；将该Label纳入长尾人工抽查和后续补题清单。"
    if risk == "P2_边界观察":
        return "保留释义，抽查边界题；结合硬负样本误收率再决定是否收窄边界。"
    return "正样本阶段可冻结；等待硬负样本验证偏宽风险。"


def evidence_note(score: float) -> str:
    if score < 0.10:
        return "按当前释义基本无关；优先核查旧ID误挂/语义漂移。"
    if score < 0.40:
        return "只有背景或邻近知识关联，不足以作为主标签。"
    if score < 0.70:
        return "存在明显关联，但当前设问不是主要考查该Label。"
    return "当前释义可覆盖，可作为对照正例。"


def choose_examples(
    results: list[dict[str, Any]], tasks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda row: (float(row["relevance_score"]), row["question_id"]))
    chosen: list[dict[str, Any]] = []
    if ordered:
        chosen.append(ordered[0])
    boundary = [row for row in ordered if 0.10 <= float(row["relevance_score"]) < 0.70]
    if boundary:
        chosen.append(boundary[len(boundary) // 2])
    positive = [row for row in ordered if float(row["relevance_score"]) >= 0.70]
    if positive:
        chosen.append(positive[-1])
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in chosen:
        task_id = str(row["task_id"])
        if task_id in seen:
            continue
        seen.add(task_id)
        output.append({**row, "task": tasks[task_id]})
    return output


def generate(args: argparse.Namespace) -> tuple[Path, Path]:
    labels = {str(row["label_id"]): row for row in read_jsonl(args.labels)}
    metrics = {str(row["label_id"]): row for row in read_jsonl(args.per_label)}
    tasks = {str(row["pair_id"]): row for row in read_jsonl(args.tasks)}
    results_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.results):
        results_by_label[str(row["label_id"])].append(row)
    strategies = (
        {str(row["label_id"]): row for row in read_jsonl(args.strategies)}
        if args.strategies and args.strategies.exists()
        else {}
    )

    if set(labels) != set(metrics):
        raise ValueError("labels and per-label metrics must cover the same Label IDs")
    if sum(int(row["completed"]) for row in metrics.values()) != len(tasks):
        raise ValueError("metric/task totals differ")

    enriched = []
    for label_id, label in labels.items():
        metric = metrics[label_id]
        strategy = strategies.get(label_id)
        risk = risk_level(metric, strategy)
        enriched.append(
            {
                "label": label,
                "metric": metric,
                "strategy": strategy,
                "risk": risk,
                "cause": cause_hypothesis(label, metric, strategy),
                "action": recommended_action(risk, label, metric),
                "examples": choose_examples(results_by_label[label_id], tasks),
            }
        )

    risk_order = {
        "P0_图谱冲突": 0,
        "P0_明显异常": 1,
        "P1_重点核验": 2,
        "L0_极端长尾": 3,
        "L1_长尾异常": 4,
        "P2_边界观察": 5,
        "L2_长尾待核": 6,
        "S_正样本稳定": 7,
    }
    enriched.sort(
        key=lambda item: (
            risk_order[item["risk"]],
            item["metric"]["match_rate"],
            item["metric"]["planned"],
            item["label"]["label_name"],
        )
    )
    counts = Counter(item["risk"] for item in enriched)
    module_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"labels": 0, "samples": 0, "match": 0, "under_55": 0, "long_tail": 0}
    )
    for item in enriched:
        parts = str(item["label"].get("label_path") or "").split("->")
        module = parts[1] if len(parts) > 1 else "未分类"
        stat = module_stats[module]
        stat["labels"] += 1
        stat["samples"] += int(item["metric"]["planned"])
        stat["match"] += int(item["metric"]["match"])
        stat["under_55"] += float(item["metric"]["match_rate"]) < 0.55
        stat["long_tail"] += int(item["metric"]["planned"]) < 300

    def pattern_stats(tokens: tuple[str, ...]) -> tuple[int, int, int]:
        items = [
            item
            for item in enriched
            if any(token in str(item["label"]["label_name"]) for token in tokens)
        ]
        return (
            len(items),
            sum(int(item["metric"]["planned"]) for item in items),
            sum(int(item["metric"]["match"]) for item in items),
        )

    full = [
        "# 高中生物458个Label正样本覆盖逐项复核",
        "",
        "## 结论摘要",
        "",
        "- 数据口径：只使用独立题，共179,568个题目—Label对，458个Label全覆盖。",
        "- 本实验检验“历史已打Label的独立题是否被当前老师释义覆盖”，历史ID只是弱监督，不是金标。",
        "- 总匹配率68.53%；这不等于31.47%的释义错误，很多反映历史误标、上位/综合Label兜底或ID语义漂移。",
        "- 正样本不能检验释义是否偏宽；所有“稳定”结论均需硬负样本二次验证。",
        "- 少于300道独立题的122个Label统一视为长尾，不根据百分比自动修改释义。",
        "",
        "## 风险分层",
        "",
        "| 分层 | Label数 | 解读 |",
        "|---|---:|---|",
    ]
    explanations = {
        "P0_图谱冲突": "Label名与释义已知冲突，需老师确认taxonomy。",
        "P0_明显异常": "样本充足且覆盖极低/大量0分，优先核验历史映射。",
        "P1_重点核验": "样本充足但覆盖偏低，需区分旧误标与释义漏项。",
        "P2_边界观察": "正样本中等，等待负样本后决定是否调整。",
        "L0_极端长尾": "1–29题，比例不稳定，应全量人工看题。",
        "L1_长尾异常": "30–299题且当前覆盖异常，长尾优先复核。",
        "L2_长尾待核": "少于300题，当前无法下自动结论。",
        "S_正样本稳定": "当前释义对历史正样本覆盖稳定，待负样本验宽度。",
    }
    for risk in risk_order:
        full.append(f"| {risk} | {counts[risk]} | {explanations[risk]} |")
    full.extend(
        [
            "",
            "## 学科模块与Label类型诊断",
            "",
            "| 学科模块 | Label数 | 题目—Label对 | 加权match | match<55% | 长尾Label |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for module, stat in sorted(module_stats.items()):
        full.append(
            f"| {module} | {stat['labels']} | {stat['samples']:,} | "
            f"{stat['match'] / stat['samples']:.1%} | {stat['under_55']} | {stat['long_tail']} |"
        )
    full.extend(
        [
            "",
            "| Label结构类型 | Label数 | 题目—Label对 | 加权match | 主要风险 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for title, tokens, note in (
        ("综合类", ("综合",), "历史上容易被当作章节兜底，而当前释义要求多模块联动。"),
        ("实验/探究/观察/调查/制作", ("实验", "探究", "观察", "调查", "制作"), "需区分实验任务与仅考相关原理。"),
        ("应用/成果/药品", ("应用", "成果", "药品"), "需区分工程原理、操作步骤与实际产品/用途。"),
    ):
        label_count, sample_count, match_count = pattern_stats(tokens)
        full.append(
            f"| {title} | {label_count} | {sample_count:,} | "
            f"{match_count / sample_count:.1%} | {note} |"
        )
    full.extend(
        [
            "",
            "其中综合类52个Label的加权match仅为50.2%，是当前最明显的系统性口径冲突：这更支持清理旧上位兜底标注，而不是把综合释义扩张为“任一子知识都命中”。",
            "",
            "## 458个Label逐项分析",
            "",
            "表内“原因假设”是基于分数分布、Label名与释义结构的筛查结论；不代替学科老师对具体题目的最终判定。",
            "",
        ]
    )
    for risk in risk_order:
        group = [item for item in enriched if item["risk"] == risk]
        if not group:
            continue
        full.extend(
            [
                f"### {risk}（{len(group)}个）",
                "",
                "| Label | ID | 样本 | match | ≥0.80 | 0分 | 原因假设 | 处理建议 | 低分证据 |",
                "|---|---|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for item in group:
            label, metric = item["label"], item["metric"]
            example = item["examples"][0] if item["examples"] else None
            evidence = (
                f"{example['question_id']}({float(example['relevance_score']):.2f})："
                f"{clip(example['task'].get('stem'), 90)}"
                if example
                else "-"
            )
            full.append(
                f"| {md(label['label_name'])} | `{label['label_id']}` | {metric['planned']} | "
                f"{pct(metric['match_rate'])} | {pct(metric['high_match_rate'])} | "
                f"{pct(metric['zero_rate'])} | {md(item['cause'])} | {md(item['action'])} | {md(evidence)} |"
            )
        full.append("")
    full.extend(
        [
            "## 总体处理决策",
            "",
            "1. 不用低匹配率直接改老师释义；先核验历史题的Label是否真的应该保留。",
            "2. 综合类Label坚持“多子模块联动”，单一子知识题不因历史挂标而并入综合Label。",
            "3. 实验类Label只在实验目的、步骤、变量、现象、误差或方案评价是作答对象时保留。",
            "4. 122个长尾Label不做自动释义调整；其中1–29题的Label应逐题复核。",
            "5. 正样本稳定不代表释义不偏宽；最终分档以正覆盖率+硬负样本误收率为准。",
            "",
        ]
    )

    focus_items = [
        item
        for item in enriched
        if item["risk"]
        in {
            "P0_图谱冲突",
            "P0_明显异常",
            "P1_重点核验",
            "L0_极端长尾",
            "L1_长尾异常",
        }
        or bool(
            ((item["strategy"] or {}).get("final_strategy") or {}).get(
                "manual_followup_required"
            )
        )
    ]
    focus = [
        "# 高中生物需重点关注的Label与释义证据",
        "",
        "## 使用说明",
        "",
        f"- 本文收录{len(focus_items)}个优先核验Label：图谱冲突、样本充足但明显异常、重点核验、极端长尾和长尾异常。",
        "- 题目来自独立题，不存在组合题父题Label并集污染。",
        "- 低分题不能直接证明释义错；它们用于请老师判定“旧标错”、“释义漏”或“该Label允许上位兜底”。",
        "- 当前只是正样本阶段；偏宽结论需等待硬负样本实验。",
        "",
        "## 先看这四类系统性问题",
        "",
        "1. **Label名—释义冲突**：“蛋白质病毒的增殖”的名称与RNA/逆转录病毒释义明显不同，需先修taxonomy。",
        "2. **综合Label口径冲突**：综合类52个Label加权match仅50.2%；大量历史题只考一个子模块，不应为迁就旧标而放宽综合释义。",
        "3. **旧ID语义/映射漂移**：“细胞质的组成与分离方法”、“基因工程药品”等在500题中出现超高0分率，且题目主题大量无关，应先清洗旧标而非改释义。",
        "4. **名称与定义层级需确认**：如“生物膜的概念与结构”的definition实际主要定义“生物膜系统”，容易与单个生物膜的结构口径错位；应由学科明确本Label的层级。",
        "",
        "## 优先级总览",
        "",
        "| 优先级 | Label数 |",
        "|---|---:|",
    ]
    for risk in risk_order:
        count = sum(item["risk"] == risk for item in focus_items)
        if count:
            focus.append(f"| {risk} | {count} |")
    focus.append("")

    for index, item in enumerate(focus_items, 1):
        label, metric = item["label"], item["metric"]
        strategy = item["strategy"] or {}
        final_strategy = strategy.get("final_strategy") or {}
        second_review = strategy.get("second_review") or {}
        focus.extend(
            [
                f"## {index}. {label['label_name']}（{item['risk']}）",
                "",
                f"- Label ID：`{label['label_id']}`",
                f"- 路径：{label.get('label_path', '')}",
                f"- 独立题样本：{metric['planned']}；match={pct(metric['match_rate'])}；≥0.80={pct(metric['high_match_rate'])}；0分={pct(metric['zero_rate'])}",
                f"- 既有策略：{final_strategy.get('mode', '-')}；人工跟进={final_strategy.get('manual_followup_required', '-')}",
                f"- 此前复核：{second_review.get('rationale', '-')}",
                f"- 原因假设：{item['cause']}",
                f"- 建议：{item['action']}",
                "",
                "### 老师当前释义",
                "",
                f"- definition：{label.get('definition', '')}",
                f"- core_concepts：{label.get('core_concepts', '')}",
                f"- common_assessments：{label.get('common_assessments', '')}",
                f"- distinctions：{label.get('distinctions', '')}",
                "",
                "### 具体题目证据",
                "",
            ]
        )
        for example in item["examples"]:
            score = float(example["relevance_score"])
            task = example["task"]
            focus.extend(
                [
                    f"#### 题目 `{example['question_id']}`，score={score:.2f}",
                    "",
                    f"- 题干：{clip(task.get('stem'), 500)}",
                    f"- 选项：{clip(task.get('options'), 420) or '-'}",
                    f"- 答案：{clip(task.get('answer_text'), 180) or '-'}",
                    f"- 解析摘要：{clip(task.get('analysis'), 520) or '-'}",
                    f"- 证据判读：{evidence_note(score)}",
                    "",
                ]
            )
        focus.extend(
            [
                "### 请学科确认",
                "",
                "1. 上述低分题是否应保留该Label？",
                "2. 若应保留，当前释义具体漏了哪种题型或知识边界？",
                "3. 若不应保留，是否可将这些历史标注作为错标清理，而不修改释义？",
                "",
            ]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "biology-458-label-positive-coverage-review.md"
    focus_path = args.output_dir / "biology-label-definition-focus-with-question-evidence.md"
    full_path.write_text("\n".join(full), encoding="utf-8")
    focus_path.write_text("\n".join(focus), encoding="utf-8")
    return full_path, focus_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--per-label", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--strategies", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    full_path, focus_path = generate(args)
    print(json.dumps({"full_review": str(full_path), "focus_review": str(focus_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
