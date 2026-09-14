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
    return _join_sections(
        _section("模式", final_strategy.get("mode")),
        _section("说明", final_strategy.get("reason") or previous_strategy.get("reason")),
        _section("自动化", final_strategy.get("automation") or previous_strategy.get("automation")),
        _section("Prompt字段", final_strategy.get("prompt_fields") or previous_strategy.get("prompt_fields")),
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
