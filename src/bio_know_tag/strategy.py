"""Per-label routing strategies for efficient biology question tagging.

The strategy is deliberately separate from the model's L1/L2/L3 category.  A
category describes the label-name audit; a strategy describes how a question
tagger should consume that label during production labeling.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from openpyxl import load_workbook

from .ds import classify_alignment
from .labels import normalize_cell


TAXONOMY_HOLD_PROMPT = "先修图谱；当前释义不足以稳定区分"
STRUCTURAL_RISK = "结构性"

REFERENCE_SHEET = "逐Label打标策略_复核"
ISSUE_SHEET = "图谱问题_复核"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _manual_review_required(category: str, stage2: dict[str, Any], issue: dict[str, Any]) -> bool:
    if category == "L3":
        return True
    if _text(issue.get("风险级别")) in {"P0", "P1", "P2"}:
        return True
    return stage2.get("audit_decision") in {
        "DS释义更准确",
        "两者都有问题",
        "无法仅凭现有信息判断",
    }


def choose_strategy(
    label: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose a deterministic production route for one label.

    ``reference`` is the optional mentor strategy workbook row and ``issue``
    is an optional row from its taxonomy-risk sheet.  Neither can silently
    override the Stage 2 result: they are retained as evidence and only add
    stricter routing when a known risk is present.
    """
    reference = reference or {}
    issue = issue or {}
    category = classify_alignment(stage2)
    risk = _text(issue.get("风险级别"))
    prompt_hint = _text(reference.get("Prompt是否需要释义"))
    code = _text(reference.get("关键词策略代码"))

    # P0/name-definition conflicts must not be sent to an automatic final
    # labeler.  This is intentionally checked before the orthogonal-axis rule.
    if risk == "P0" or prompt_hint == TAXONOMY_HOLD_PROMPT:
        mode = "taxonomy_hold"
        automation = "暂停自动最终打标"
        reason = "Label 名与释义或兄弟节点存在不可稳定裁决的图谱冲突，先修订 taxonomy。"
    # KM labels describe information form, source stage, context, or ability,
    # not the biology concept itself.  They should be emitted as a separate
    # dimension even when the name audit is poor.
    elif risk == STRUCTURAL_RISK or code == "KM" or "维度定义" in prompt_hint:
        mode = "separate_dimension"
        automation = "作为独立维度预测，不写入知识标签"
        reason = "这是信息载体/能力/来源等正交维度，和知识 Label 分开存储。"
    elif risk in {"P1", "P2"} or category == "L3":
        mode = "strict_definition"
        automation = "提供老师边界后由LLM最终裁决"
        reason = "存在已知边界风险或 Stage 2 对齐不足，必须提供命中条件与排除条件。"
    elif category == "L2" or stage2.get("name_sufficiency") == "需要原释义":
        mode = "compact_definition"
        automation = "提供精简老师释义后由LLM裁决"
        reason = "名称能表达主题但不足以稳定决定题目归属，补充精简定义和易混淆边界。"
    elif "建议给精简释义" in prompt_hint or "必须给精简释义" in prompt_hint:
        mode = "compact_definition"
        automation = "提供精简老师释义后由LLM裁决"
        reason = "参考策略要求任务形态或命中/排除边界，补充精简老师释义以降低误标。"
    elif "可不给长释义" in prompt_hint:
        mode = "name_plus_boundary"
        automation = "名称召回并附一条边界后由LLM裁决"
        reason = "名称基本可理解；只附老师的易混淆边界，兼顾吞吐和边界稳定性。"
    else:
        mode = "name_only"
        automation = "名称召回后由LLM按最小充分知识集裁决"
        reason = "Stage 2 判断名称本身足够，先用名称高效召回，再由设问语义做最终裁决。"

    manual = _manual_review_required(category, stage2, issue)
    if mode == "taxonomy_hold":
        manual = True
    if mode == "separate_dimension":
        manual = False

    return {
        "mode": mode,
        "automation": automation,
        "manual_review_required": manual,
        "category": category,
        "strategy_code": code or None,
        "reason": reason,
        "reference_prompt_hint": prompt_hint or None,
        "known_risk_level": risk or None,
        "teacher_definition_required": mode in {"compact_definition", "strict_definition"},
        "keep_stage2_evidence": True,
        "stage1_used_for_routing": bool(stage1),
    }


def strategy_reason(strategy: dict[str, Any], category: str | None = None) -> str:
    """Return a compact human-readable audit reason for reports and tables."""
    category = category or _text(strategy.get("category")) or "unknown"
    suffix = "需要人工复核" if strategy.get("manual_review_required") else "可进入自动流程"
    return f"{category}；{strategy.get('reason', '')}{suffix}。"


def latest_successful(records: Iterable[dict[str, Any]], id_field: str = "label_id") -> dict[str, dict[str, Any]]:
    """Return the last successful parsed record for each identifier."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("error") or not isinstance(record.get("parsed_response"), dict):
            continue
        latest[str(record[id_field])] = record
    return latest


def load_reference_workbook(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load optional mentor strategies and taxonomy issues keyed by label.

    The reference workbook is advisory.  Its rows are preserved in the
    generated ledger so a reviewer can see exactly which recommendation was
    used to select a route.
    """
    workbook = load_workbook(Path(path), read_only=True, data_only=True)
    if REFERENCE_SHEET not in workbook.sheetnames:
        raise ValueError(f"missing sheet: {REFERENCE_SHEET}")
    strategy_sheet = workbook[REFERENCE_SHEET]
    strategy_rows = list(strategy_sheet.iter_rows(values_only=True))
    if not strategy_rows:
        raise ValueError(f"empty sheet: {REFERENCE_SHEET}")
    headers = [normalize_cell(value) for value in strategy_rows[0]]
    id_index = headers.index("末级知识点编号")
    strategy_by_id: dict[str, dict[str, Any]] = {}
    for row in strategy_rows[1:]:
        values = {
            headers[index]: normalize_cell(row[index] if index < len(row) else None)
            for index in range(len(headers))
            if headers[index]
        }
        label_id = normalize_cell(row[id_index] if id_index < len(row) else None)
        if not label_id:
            continue
        if label_id in strategy_by_id:
            raise ValueError(f"duplicate reference label id: {label_id}")
        strategy_by_id[label_id] = values

    issue_by_name: dict[str, dict[str, Any]] = {}
    if ISSUE_SHEET in workbook.sheetnames:
        issue_sheet = workbook[ISSUE_SHEET]
        issue_rows = list(issue_sheet.iter_rows(values_only=True))
        if issue_rows:
            issue_headers = [normalize_cell(value) for value in issue_rows[0]]
            name_index = issue_headers.index("Label")
            for row in issue_rows[1:]:
                issue = {
                    issue_headers[index]: normalize_cell(row[index] if index < len(row) else None)
                    for index in range(len(issue_headers))
                    if issue_headers[index]
                }
                label_name = normalize_cell(row[name_index] if name_index < len(row) else None)
                if not label_name:
                    continue
                if label_name in issue_by_name:
                    raise ValueError(f"duplicate reference issue label: {label_name}")
                issue_by_name[label_name] = issue
    return strategy_by_id, issue_by_name


def _prompt_fields(mode: str) -> list[str]:
    if mode == "name_only":
        return ["label_name"]
    if mode == "name_plus_boundary":
        return ["label_name", "distinctions"]
    if mode == "compact_definition":
        return ["label_name", "definition", "core_concepts", "distinctions"]
    if mode == "strict_definition":
        return [
            "label_name",
            "definition",
            "core_concepts",
            "common_assessments",
            "distinctions",
            "include_rule",
            "exclude_rule",
        ]
    if mode == "separate_dimension":
        return ["label_name", "dimension_definition", "knowledge_label_relation"]
    return []


def build_strategy_records(
    labels: Iterable[dict[str, Any]],
    stage1_by_id: dict[str, dict[str, Any]],
    stage2_by_id: dict[str, dict[str, Any]],
    *,
    reference_by_id: dict[str, dict[str, Any]] | None = None,
    issues_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join taxonomy, both DS stages, and one routing decision per label."""
    reference_by_id = reference_by_id or {}
    issues_by_name = issues_by_name or {}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in labels:
        label_id = str(label["label_id"])
        if label_id in seen:
            raise ValueError(f"duplicate label id: {label_id}")
        seen.add(label_id)
        if label_id not in stage1_by_id:
            raise ValueError(f"missing successful Stage 1 evidence: {label_id}")
        if label_id not in stage2_by_id:
            raise ValueError(f"missing successful Stage 2 evidence: {label_id}")
        stage1_record = stage1_by_id[label_id]
        stage2_record = stage2_by_id[label_id]
        stage1 = stage1_record["parsed_response"]
        stage2 = stage2_record["parsed_response"]
        calculated_category = classify_alignment(stage2)
        recorded_category = stage2_record.get("category")
        if recorded_category and recorded_category != calculated_category:
            raise ValueError(
                f"Stage 2 category mismatch for {label_id}: "
                f"recorded={recorded_category} calculated={calculated_category}"
            )
        reference = reference_by_id.get(label_id, {})
        issue = issues_by_name.get(label["label_name"], {})
        strategy = choose_strategy(
            label,
            stage1,
            stage2,
            reference=reference,
            issue=issue,
        )
        strategy["prompt_fields"] = _prompt_fields(strategy["mode"])
        strategy["audit_reason"] = strategy_reason(strategy, calculated_category)
        output.append(
            {
                **label,
                "stage1": stage1,
                "stage2_judge": stage2,
                "stage2_category": calculated_category,
                "stage1_provenance": {
                    "created_at": stage1_record.get("created_at"),
                    "endpoint": stage1_record.get("endpoint"),
                    "attempts": stage1_record.get("attempts"),
                },
                "stage2_provenance": {
                    "created_at": stage2_record.get("created_at"),
                    "endpoint": stage2_record.get("endpoint"),
                    "attempts": stage2_record.get("attempts"),
                },
                "reference_strategy": reference or None,
                "taxonomy_issue": issue or None,
                "strategy": strategy,
            }
        )
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_strategy_outputs(
    records: list[dict[str, Any]],
    output_path: str | Path,
    report_path: str | Path,
    *,
    input_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Write deterministic JSONL and a compact coverage report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(output)

    mode_counts = Counter(record["strategy"]["mode"] for record in records)
    category_counts = Counter(record["stage2_category"] for record in records)
    report: dict[str, Any] = {
        "input": len(records),
        "processed": len(records),
        "error": 0,
        "stage2_category_counts": dict(sorted(category_counts.items())),
        "strategy_mode_counts": dict(sorted(mode_counts.items())),
        "manual_review_count": sum(
            bool(record["strategy"]["manual_review_required"]) for record in records
        ),
        "taxonomy_issue_count": sum(bool(record.get("taxonomy_issue")) for record in records),
        "reference_strategy_count": sum(bool(record.get("reference_strategy")) for record in records),
    }
    if input_paths:
        report["input_sha256"] = {
            name: sha256_file(path) for name, path in sorted(input_paths.items())
        }
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = report_output.with_name(f".{report_output.name}.tmp")
    report_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_temporary.replace(report_output)
    return report
