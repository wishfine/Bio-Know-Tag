"""Export the per-label DS/GPT review ledger as a Markdown document."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MARKDOWN_HEADERS = [
    "序号",
    "Label ID",
    "Label名称",
    "Label路径",
    "原释义",
    "DS释义",
    "DS Judge结果",
    "GPT Judge结果（二次复核）",
    "处理策略",
    "风险/备注",
]


def _plain(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(f"{index}. {item}" for index, item in enumerate(value, 1))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _section(title: str, value: Any) -> str:
    rendered = _plain(value)
    return f"{title}：{rendered}" if rendered else ""


def _join_sections(*sections: str) -> str:
    return "\n".join(section for section in sections if section)


def _md_cell(value: Any) -> str:
    """Make a value safe for one Markdown table cell."""

    text = _plain(value).replace("|", "\\|")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>") or "—"


def _teacher_definition(record: dict[str, Any]) -> str:
    return _join_sections(
        _section("定义", record.get("definition")),
        _section("核心概念", record.get("core_concepts")),
        _section("常见考查", record.get("common_assessments")),
        _section("易混淆区分", record.get("distinctions")),
    )


def _ds_definition(record: dict[str, Any]) -> str:
    stage1 = record.get("stage1") or {}
    return _join_sections(
        _section("核心含义", stage1.get("core_meaning")),
        _section("包含内容", stage1.get("included_content")),
        _section("排除内容", stage1.get("excluded_content")),
    )


def _ds_judge(record: dict[str, Any]) -> str:
    judge = record.get("stage2_judge") or {}
    return _join_sections(
        _section("类别", record.get("stage2_category")),
        _section("分数", judge.get("alignment_score")),
        _section("名称充分性", judge.get("name_sufficiency")),
        _section("结论", judge.get("audit_decision")),
        _section("审核理由", judge.get("audit_reason")),
        _section("边界差异", judge.get("boundary_differences")),
        _section("遗漏", judge.get("omissions")),
        _section("扩展", judge.get("expansions")),
    )


def _gpt_judge(record: dict[str, Any]) -> str:
    review = record.get("second_review") or {}
    final_strategy = record.get("final_strategy") or {}
    return _join_sections(
        _section("状态", review.get("status")),
        _section("置信度", review.get("confidence")),
        _section("此前策略", review.get("previous_mode")),
        _section("最终策略", review.get("final_mode") or final_strategy.get("mode")),
        _section("是否调整", review.get("adjusted_from_previous")),
        _section("复核结论", review.get("rationale")),
        _section("人工跟进", review.get("manual_followup_required")),
    )


def _strategy(record: dict[str, Any]) -> str:
    final_strategy = record.get("final_strategy") or {}
    previous_strategy = record.get("strategy") or {}
    review = record.get("second_review") or {}
    return _join_sections(
        _section("模式", final_strategy.get("mode")),
        _section("说明", final_strategy.get("reason") or previous_strategy.get("reason")),
        _section("自动化", final_strategy.get("automation") or previous_strategy.get("automation")),
        _section("Prompt字段", final_strategy.get("prompt_fields") or previous_strategy.get("prompt_fields")),
        _section(
            "业务边界",
            final_strategy.get("operational_boundary")
            or review.get("operational_boundary"),
        ),
        _section(
            "需要老师释义",
            final_strategy.get("teacher_definition_required")
            if "teacher_definition_required" in final_strategy
            else previous_strategy.get("teacher_definition_required"),
        ),
        _section("置信度", final_strategy.get("confidence") or previous_strategy.get("confidence")),
    )


def _risk(record: dict[str, Any]) -> str:
    issue = record.get("taxonomy_issue")
    review = record.get("second_review") or {}
    strategy = record.get("final_strategy") or {}
    if isinstance(issue, dict):
        issue_text = _join_sections(
            _section("taxonomy风险级别", issue.get("风险级别") or issue.get("risk_level")),
            _section("问题", issue.get("问题") or issue.get("problem")),
            _section("建议", issue.get("建议") or issue.get("suggestion")),
        )
    else:
        issue_text = _plain(issue)
    known_risk = strategy.get("known_risk_level") or review.get("known_risk_level")
    return _join_sections(_section("已知风险", known_risk), issue_text)


def build_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        rows.append(
            {
                "序号": index,
                "Label ID": record.get("label_id", ""),
                "Label名称": record.get("label_name", ""),
                "Label路径": record.get("label_path", ""),
                "原释义": _teacher_definition(record),
                "DS释义": _ds_definition(record),
                "DS Judge结果": _ds_judge(record),
                "GPT Judge结果（二次复核）": _gpt_judge(record),
                "处理策略": _strategy(record),
                "风险/备注": _risk(record),
            }
        )
    return rows


def _summary(records: list[dict[str, Any]]) -> list[str]:
    categories = Counter(str(r.get("stage2_category", "")) for r in records)
    scores = Counter(str((r.get("stage2_judge") or {}).get("alignment_score", "")) for r in records)
    statuses = Counter(str((r.get("second_review") or {}).get("status", "")) for r in records)
    modes = Counter(str((r.get("final_strategy") or {}).get("mode", "")) for r in records)
    reviews = [r.get("second_review") or {} for r in records]
    return [
        f"- Label总数：{len(records)}",
        f"- DS Judge类别：{dict(sorted(categories.items()))}",
        f"- DS Judge分数：{dict(sorted(scores.items()))}",
        f"- GPT Judge状态：{dict(sorted(statuses.items()))}",
        f"- 最终处理策略：{dict(sorted(modes.items()))}",
        f"- GPT Judge调整数：{sum(bool(r.get('adjusted_from_previous')) for r in reviews)}",
        f"- 需要人工跟进：{sum(bool(r.get('manual_followup_required')) for r in reviews)}",
    ]


def _rules_section(records: list[dict[str, Any]]) -> list[str]:
    statuses = Counter(str((record.get("second_review") or {}).get("status", "")) for record in records)
    modes = Counter(str((record.get("final_strategy") or {}).get("mode", "")) for record in records)
    return [
        "## L1 / L2 / L3 是什么",
        "",
        "这三个等级描述的是‘Label 名称本身’与老师原释义的对齐情况，不是题目打标准确率：",
        "",
        "- **L1**：名称本身基本足够理解；通常为对齐分数 4–5，且 Judge 没有判定为实质冲突。",
        "- **L2**：主题方向基本正确，但仅看名称不足以稳定决定题目归属，需要带上原释义或精简边界。",
        "- **L3**：存在重要遗漏/扩张、明显方向偏差或 taxonomy 问题，需要严格释义或暂停自动化。",
        "",
        "代码中的实际归类规则是：分数 ≤3、Judge 结论为‘DS释义更准确/两者都有问题/无法仅凭现有信息判断’，或名称充分性为‘标签体系有问题’时归为 L3；否则名称充分性为‘需要原释义’时归为 L2；其余归为 L1。",
        "",
        "## DS 释义 Prompt",
        "",
        "Stage1 只把 Label 名称发给 DS，不发送老师原释义。系统消息和用户 Prompt 如下（`{label_name}` 为实际 Label 名）：",
        "",
        "```text",
        "system: 你是严谨的高中生物知识点判别器。",
        "",
        "user:",
        "你是一名高中生物教师。",
        "",
        "现在给你一个高中生物知识点标签：",
        "【{label_name}】",
        "",
        "在不知道任何已有知识点释义的情况下，仅根据标签名称和你的高中生物知识，写出你认为这个标签对应的知识范围。",
        "",
        "请只输出一个 JSON 对象，字段严格如下：",
        '{"core_meaning":"核心含义","included_content":["应该包含的考查内容"],"excluded_content":["不应该包含的相近内容"]}',
        "",
        "不要猜测标签体系设计者的特殊规则。不要输出 Markdown 或 JSON 之外的文字。",
        "```",
        "",
        "请求参数固定为 `temperature=0`、`max_tokens=1024`，模型默认是 `DeepSeek-V4-Flash`。",
        "",
        "## DS Judge Prompt 与分数",
        "",
        "Stage2 把老师四列原释义与 Stage1 的 DS 释义一起发给 DS Judge。它要求先逐项检查遗漏、扩张和边界，再输出结构化结果：",
        "",
        "```text",
        "你是一名严谨的高中生物知识点体系审核专家。",
        "",
        "请比较同一 Label 的老师原释义与‘仅看 Label 名’生成的 DS 释义。不要默认任一方必然正确。",
        "",
        "Label 名称：{label_name}",
        "老师原释义：{definition, core_concepts, common_assessments, distinctions}",
        "DS 生成释义：{core_meaning, included_content, excluded_content}",
        "",
        "对齐分标准：",
        "5 基本完全一致；4 核心一致，仅边界有少量差异；3 主体一致但有重要缺失或扩张；2 理解方向明显偏差；1 基本不是同一知识点。",
        "",
        "请输出 JSON：alignment_score、omissions、expansions、boundary_differences、audit_decision、audit_reason、name_sufficiency。",
        "其中 name_sufficiency 必须回答：只给名称能否稳定决定题目是否属于该 Label。",
        "```",
        "",
        "因此，DS Judge 分数衡量的是‘DS 释义是否贴合老师 taxonomy’，不是 DS 对题目打标的 Accuracy。分数低不一定等于老师一定正确，Prompt 明确要求不要默认任一方必然正确。",
        "",
        "## GPT Judge 状态",
        "",
        "文档中的 GPT Judge 指 `second_review` 二次复核字段。它不是另一份原始 DS 证据，而是基于老师释义、Stage1、Stage2、参考策略和 taxonomy 风险对上一版路由做的独立复核：",
        "",
        "- **confirmed**：逐条核对后确认原策略，无需额外人工跟进。",
        "- **confirmed_with_caution**：总体路由可用，但存在边界风险；当前统一收紧为 `strict_definition`，需要人工跟进。",
        f"- **adjusted**：二次复核发现原策略可能造成实质误标，已调整最终策略；本批共 {statuses.get('adjusted', 0)} 个。",
        f"- **routed_separately**：这是题型/数据形式/能力等正交维度，不写入知识 Label，单独预测；本批共 {statuses.get('routed_separately', 0)} 个。",
        f"- **taxonomy_hold**：Label 与释义或兄弟节点无法稳定区分，暂停自动最终打标，先修 taxonomy；本批共 {statuses.get('taxonomy_hold', 0)} 个。",
        "",
        "## 最终处理策略",
        "",
        "- **name_only**：只用 Label 名做候选召回，再由 LLM 根据设问判断；本批最终数量为 0。",
        f"- **name_plus_boundary**：Label 名 + 老师的一条易混淆边界；兼顾吞吐与边界稳定性，本批 {modes.get('name_plus_boundary', 0)} 个。",
        f"- **compact_definition**：Label 名 + 老师定义/核心概念/易混淆区分等精简释义，本批 {modes.get('compact_definition', 0)} 个。",
        f"- **strict_definition**：提供完整老师释义，并明确命中条件与排除条件；用于 L3 或 P1/P2/高风险边界，本批 {modes.get('strict_definition', 0)} 个。",
        f"- **separate_dimension**：信息载体、题型、能力、来源等结构性维度与知识标签分开存储，本批 {modes.get('separate_dimension', 0)} 个。",
        f"- **taxonomy_hold**：不让模型猜，先修订 taxonomy 或使用来源字段硬路由，本批 {modes.get('taxonomy_hold', 0)} 个。",
        "",
    ]


def _manual_followup_section(records: list[dict[str, Any]]) -> list[str]:
    selected = [
        record
        for record in records
        if (record.get("second_review") or {}).get("manual_followup_required")
    ]
    lines = [
        "## 需要人工跟进的 Label",
        "",
        f"共 **{len(selected)}** 个。人工跟进不代表整条记录失败，而是要求在正式全量自动打标前，对这些 Label 的边界、来源或 taxonomy 做一次确认。",
        "",
        "| 序号 | Label名称 | label_id | DS Judge | 分数 | GPT状态 | 最终策略 | 人工跟进原因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, record in enumerate(selected, 1):
        stage2 = record.get("stage2_judge") or {}
        review = record.get("second_review") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _md_cell(record.get("label_name")),
                    _md_cell(record.get("label_id")),
                    _md_cell(record.get("stage2_category")),
                    _md_cell(stage2.get("alignment_score")),
                    _md_cell(review.get("status")),
                    _md_cell(review.get("final_mode")),
                    _md_cell(review.get("rationale")),
                ]
            )
            + " |"
        )
    return lines


def build_markdown(records: Iterable[dict[str, Any]]) -> str:
    source_records = list(records)
    rows = build_rows(source_records)
    lines = [
        "# 高中生物 458 个 Label：原释义、DS/GPT Judge 与处理策略",
        "",
        "> 本文由 `configs/label_strategies.review2.jsonl` 导出，每行一个 Label。",
        "> `DS Judge` 是第一阶段 DS 释义与老师原释义的对齐审核；`GPT Judge（二次复核）` 是基于老师释义、两轮证据和参考策略的逐条复核结果。",
        "> 表格中列表项使用 `<br>` 换行；建议用 Markdown 编辑器或浏览器查看，并通过页面搜索定位 Label。",
        "",
        "## 统计摘要",
        "",
        *_summary(source_records),
        "",
        *_rules_section(source_records),
        "## 逐 Label 复核表",
        "",
        "| " + " | ".join(MARKDOWN_HEADERS) + " |",
        "| " + " | ".join("---" for _ in MARKDOWN_HEADERS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row[header]) for header in MARKDOWN_HEADERS) + " |")
    lines.extend(
        [
            "",
            *_manual_followup_section(source_records),
            "",
            "## 字段说明",
            "",
            "- **原释义**：老师图谱的定义、核心概念、常见考查和易混淆区分。",
            "- **DS释义**：Stage1 DS 输出的核心含义、包含内容和排除内容。",
            "- **DS Judge结果**：Stage2 的 L1/L2/L3、对齐分数、名称充分性、结论及理由。",
            "- **GPT Judge结果（二次复核）**：第二轮逐条复核的状态、置信度、策略调整和人工跟进标记。",
            "- **处理策略**：`name_plus_boundary`、`compact_definition`、`strict_definition`、`separate_dimension` 或 `taxonomy_hold` 等最终路由。",
            "- **风险/备注**：已知风险以及参考策略表中的 taxonomy 问题；空白表示没有额外风险记录。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown(records: Iterable[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    output.write_text(build_markdown(rows), encoding="utf-8")
    return {"output": str(output), "rows": len(rows)}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
