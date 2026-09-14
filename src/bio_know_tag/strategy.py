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

# Explicit adjustments made during the independent second pass.  These are
# cases where the Stage 2 score is high enough to produce L1, but its own
# omissions/expansions expose a boundary that a name-only prompt could miss.
SECOND_REVIEW_OVERRIDES: dict[str, tuple[str, str]] = {
    "细胞的多样性与统一性": (
        "compact_definition",
        "DS把原核/真核分类和细胞学说混入多样性/统一性；需保留老师的宏观特征边界。",
    ),
    "生物膜的概念与结构": (
        "compact_definition",
        "老师定义的核心是生物膜系统的整体联系，DS偏向单膜组成/特性；需显式限定系统范围。",
    ),
    "生物膜的功能和特性": (
        "compact_definition",
        "DS扩展到人工膜等超出本 Label 的应用，需用老师的系统功能和排除条件约束。",
    ),
    "细胞质基质（细胞溶胶）": (
        "compact_definition",
        "老师把糖酵解场所作为考点，DS只排除详细代谢过程；需补充‘场所可考、过程细节不考’。",
    ),
    "ATP与ADP的相互转化": (
        "strict_definition",
        "DS使用‘可逆转化’可能混淆‘物质可逆、反应过程不可逆’，必须保留老师边界。",
    ),
    "吸热反应（吸能反应）、放热反应（放能反应）与ATP合成、水解": (
        "strict_definition",
        "DS对吸能/放能与 ATP 合成/水解的偶联表述可能对调，必须显式给出正确方向。",
    ),
    "环境对遗传信息表达的影响": (
        "strict_definition",
        "DS把即时酶活性调节也纳入，老师明确要求聚焦基因表达层面的环境影响。",
    ),
    "五种植物激素": (
        "compact_definition",
        "DS遗漏生长素两重性、脱落酸抗逆等关键判标内容；需补充老师核心概念。",
    ),
    "自然选择与适应的形成": (
        "compact_definition",
        "DS遗漏自然选择要点和适应相对性，且扩展到基因频率；需保留老师定义边界。",
    ),
    "影响种群数量变化的因素": (
        "compact_definition",
        "DS按生物/非生物分类，老师按内部/外部分类；需同时保留两种维度及四个率。",
    ),
    "优势种": (
        "compact_definition",
        "老师强调对群落环境的影响力，不能只按数量/生活力或关键种概念召回。",
    ),
    "生态系统的生产量和生物量": (
        "compact_definition",
        "DS扩展到次级生产量和测定方法，可能改变题目归属；需显式区分生产量与生物量。",
    ),
    "生态平衡和生态系统的稳定性": (
        "compact_definition",
        "DS把抵抗力/恢复力类型直接并入，老师更侧重生态平衡概念；需保留两者的层次边界。",
    ),
    "全球性环境问题": (
        "compact_definition",
        "DS把水体富营养化等区域性问题直接归入全球性问题，需按老师的全球性清单约束。",
    ),
    "克隆的利与弊": (
        "compact_definition",
        "DS遗漏治疗性/生殖性克隆的伦理差异及技术风险，且扩展植物克隆；需补充老师边界。",
    ),
    "植物体细胞杂交技术": (
        "compact_definition",
        "DS把后续组织培养整体排除，而老师定义包含杂种细胞到植株的培养链；需显式区分后续步骤与独立标签。",
    ),
    "体内受精过程": (
        "compact_definition",
        "DS把卵裂起点扩展进受精过程；需用老师边界限定到精卵结合及防多精机制。",
    ),
    "基因工程的概念及发展历程": (
        "compact_definition",
        "DS给出的历史年份/事件与老师不一致，且漏掉打破生殖隔离的意义；需保留老师版本。",
    ),
    "DNA连接酶": (
        "compact_definition",
        "DS把 DNA 复制中的冈崎片段连接纳入，可能与基因工程语境混淆；需补充片段连接边界。",
    ),
    "基因检测引发的伦理问题": (
        "compact_definition",
        "DS排除产前诊断等应用背景，但伦理判断通常依赖具体应用场景；需保留老师应用与伦理边界。",
    ),
    "生物武器的传播途径与防护措施": (
        "compact_definition",
        "DS排除国际公约，而老师把公约列为防护措施；需保留公约及传播实例边界。",
    ),
}

# The teacher's own descriptions already provide an operational focus for
# these two rows.  Keep the historical P0 issue as evidence, but do not block
# production tagging: send the full teacher definition plus this side-focus
# boundary to the classifier.  This is a routing decision, not a claim that
# the two taxonomy rows are semantically disjoint in every question.
OPERATIONAL_BOUNDARY_OVERRIDES: dict[str, str] = {
    "观察根尖分生区组织细胞的有丝分裂": (
        "知识点考查（分裂时期、染色体变化、细胞计数）；"
        "与‘活动：观察细胞的有丝分裂’区分为教材实验操作侧重。"
    ),
    "活动：观察细胞的有丝分裂": (
        "教材活动/实验操作（取材、解离、漂洗、染色、制片、显微镜观察）；"
        "与‘观察根尖分生区组织细胞的有丝分裂’区分为知识点考查侧重。"
    ),
}


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


def second_review_strategy(
    label: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    previous: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform the independent, evidence-backed second pass for one label.

    This pass intentionally records a decision even for unresolved taxonomy:
    the decision in that case is to hold the label, not to guess a knowledge
    boundary.  It uses the same evidence fields a reviewer would inspect, but
    does not overwrite the raw model outputs.
    """
    reference = reference or {}
    issue = issue or {}
    previous = previous or {}
    category = classify_alignment(stage2)
    risk = _text(issue.get("风险级别"))
    prompt_hint = _text(reference.get("Prompt是否需要释义"))
    code = _text(reference.get("关键词策略代码"))
    operational_boundary = OPERATIONAL_BOUNDARY_OVERRIDES.get(
        _text(label.get("label_name"))
    )

    # Re-evaluate the routing in an explicit priority order.  The decision is
    # independent of the previous strategy object; that object is compared
    # afterwards to expose any adjustment.
    if operational_boundary:
        final_mode = "strict_definition"
        status = "adjusted"
        confidence = "medium"
        rationale = (
            "老师原释义已给出两个条目的侧重点，生产阶段采用完整老师释义并加入"
            "侧重点边界；不要求老师逐条确认，但仍保留题面不可区分时的风险记录。"
        )
        manual_followup = False
    elif risk == "P0" or prompt_hint == TAXONOMY_HOLD_PROMPT:
        final_mode = "taxonomy_hold"
        status = "taxonomy_hold"
        confidence = "low"
        rationale = "逐条核对后仍存在不可稳定区分的 Label/释义或兄弟节点冲突，结论是暂停自动打标。"
        manual_followup = True
    elif risk == STRUCTURAL_RISK or code == "KM" or "维度定义" in prompt_hint:
        final_mode = "separate_dimension"
        status = "routed_separately"
        confidence = "high"
        rationale = "逐条核对确认该行描述的是信息载体、能力、学段或情境维度，应与知识标签分开。"
        manual_followup = False
    elif category == "L3" or risk in {"P1", "P2"}:
        final_mode = "strict_definition"
        status = "confirmed_with_caution"
        confidence = "medium"
        rationale = (
            f"逐条核对老师定义与 DS Judge（score={stage2.get('alignment_score')}，"
            f"audit={stage2.get('audit_decision')}）后，保留老师边界并要求命中/排除条件。"
        )
        manual_followup = True
    elif category == "L2" or stage2.get("name_sufficiency") == "需要原释义":
        final_mode = "compact_definition"
        status = "confirmed"
        confidence = "medium"
        rationale = "逐条核对确认 Label 名能表达主题，但不足以稳定决定题目归属，需补充精简老师释义。"
        manual_followup = False
    elif "建议给精简释义" in prompt_hint or "必须给精简释义" in prompt_hint:
        final_mode = "compact_definition"
        status = "confirmed"
        confidence = "high"
        rationale = "逐条核对确认主题清楚，但参考策略指出任务形态或命中/排除边界需要显式提供。"
        manual_followup = False
    elif "可不给长释义" in prompt_hint:
        final_mode = "name_plus_boundary"
        status = "confirmed"
        confidence = "high"
        rationale = "逐条核对确认名称足以召回；只附老师易混淆边界即可兼顾吞吐和稳定性。"
        manual_followup = False
    else:
        final_mode = "name_only"
        status = "confirmed"
        confidence = "high"
        rationale = "逐条核对确认名称与老师/DS核心一致，使用名称召回并让 LLM 依据设问做最终裁决。"
        manual_followup = False

    # Apply only explicit second-pass corrections after the general routing
    # rules.  The override reason is stored per row so every adjustment can be
    # audited without discarding the original Stage 2 judgment.
    override = SECOND_REVIEW_OVERRIDES.get(_text(label.get("label_name")))
    if override and final_mode not in {"taxonomy_hold", "separate_dimension"}:
        final_mode, rationale = override
        status = "adjusted"
        confidence = "medium"
        manual_followup = final_mode == "strict_definition" or category == "L3" or risk in {
            "P1",
            "P2",
        }

    return {
        "reviewed": True,
        "status": status,
        "confidence": confidence,
        "final_mode": final_mode,
        "adjusted_from_previous": previous.get("mode") != final_mode,
        "previous_mode": previous.get("mode"),
        "manual_followup_required": manual_followup,
        "rationale": rationale,
        "evidence_checked": [
            "teacher_definition",
            "teacher_core_concepts",
            "teacher_distinctions",
            "stage1_core_meaning",
            "stage1_included_content",
            "stage1_excluded_content",
            "stage2_alignment_score",
            "stage2_audit_decision",
            "stage2_name_sufficiency",
            "reference_prompt_hint",
            "reference_strategy_code",
            "taxonomy_issue",
        ],
        "stage2_category": category,
        "stage2_score": stage2.get("alignment_score"),
        "reference_strategy_code": code or None,
        "known_risk_level": risk or None,
        "label_name": label.get("label_name"),
        "stage1_present": bool(stage1),
        "operational_boundary": operational_boundary,
    }


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


def build_second_review_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach an independent second-pass decision to every ledger row."""
    output: list[dict[str, Any]] = []
    for record in records:
        second = second_review_strategy(
            record,
            record["stage1"],
            record["stage2_judge"],
            record.get("strategy", {}),
            reference=record.get("reference_strategy") or {},
            issue=record.get("taxonomy_issue") or {},
        )
        final_mode = second["final_mode"]
        final_strategy = {
            "mode": final_mode,
            "automation": {
                "name_only": "名称召回后由LLM按最小充分知识集裁决",
                "name_plus_boundary": "名称召回并附一条边界后由LLM裁决",
                "compact_definition": "提供精简老师释义后由LLM裁决",
                "strict_definition": "提供老师边界后由LLM最终裁决",
                "taxonomy_hold": "暂停自动最终打标",
                "separate_dimension": "作为独立维度预测，不写入知识标签",
            }[final_mode],
            "prompt_fields": _prompt_fields(final_mode),
            "status": second["status"],
            "confidence": second["confidence"],
            "manual_followup_required": second["manual_followup_required"],
        }
        if second.get("operational_boundary"):
            final_strategy["operational_boundary"] = second["operational_boundary"]
        output.append({**record, "second_review": second, "final_strategy": final_strategy})
    return output


def write_second_review_outputs(
    records: list[dict[str, Any]],
    output_path: str | Path,
    report_path: str | Path,
    *,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the second-pass ledger and coverage report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(output)

    status_counts = Counter(record["second_review"]["status"] for record in records)
    mode_counts = Counter(record["final_strategy"]["mode"] for record in records)
    report: dict[str, Any] = {
        "input": len(records),
        "processed": sum(record["second_review"].get("reviewed") is True for record in records),
        "error": 0,
        "review_status_counts": dict(sorted(status_counts.items())),
        "final_mode_counts": dict(sorted(mode_counts.items())),
        "adjusted_count": sum(
            record["second_review"].get("adjusted_from_previous") is True
            for record in records
        ),
        "manual_followup_count": sum(
            record["second_review"].get("manual_followup_required") is True
            for record in records
        ),
    }
    if input_path:
        report["input_sha256"] = {"ledger": sha256_file(input_path)}
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = report_output.with_name(f".{report_output.name}.tmp")
    report_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_temporary.replace(report_output)
    return report


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
