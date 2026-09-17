"""DS adjudication over a small hybrid candidate set."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.ds import DSRequestError, append_evidence, parse_json_content
from bio_know_tag.retrieval import format_label_path


PROMPT_VERSION = "candidate-adjudication-v10.1-target-aligned-hard-gates"
CANDIDATE_ORDER_VERSION = "candidate-adjudication-v8.3-internal-reflection"
EVIDENCE_SOURCES = ("parent_stem", "stem", "options", "answer_text", "analysis")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            records.append(value)
    return records


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_evidence_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).casefold()


def _ensure_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("run manifest mismatch; use a new run directory")
        return
    _write_json_atomic(path, manifest)


def build_adjudication_prompt(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    *,
    diagnostic_focus_label_ids: set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    question_id = str(unit.get("question_id") or "")
    shuffled_candidates = sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{question_id}\0{candidate['label_id']}\0{CANDIDATE_ORDER_VERSION}".encode(
                "utf-8"
            )
        ).digest(),
    )
    code_map = {
        f"C{index:02d}": str(candidate["label_id"])
        for index, candidate in enumerate(shuffled_candidates, 1)
    }
    candidate_cards = []
    for code, label_id in code_map.items():
        label = labels_by_id[label_id]
        candidate_cards.append(
            {
                "code": code,
                "label_name": label.get("label_name", ""),
                "label_path": format_label_path(label.get("label_path")),
                "definition": label.get("definition", ""),
                "core_concepts": label.get("core_concepts", ""),
                "distinctions": label.get("distinctions", ""),
            }
        )
    question = {
        "question_id": question_id,
        "unit_type": unit.get("unit_type", ""),
        "parent_stem": str(unit.get("parent_stem") or "")[:3000],
        "stem": str(unit.get("stem") or "")[:5000],
        "options": str(unit.get("options") or "")[:3000],
        "answer_text": str(unit.get("answer_text") or "")[:2000],
        "analysis": str(unit.get("analysis") or "")[:6000],
        "parent_context_missing": bool(
            (unit.get("flags") or {}).get("parent_context_missing")
        ),
        "image_context_missing": bool(
            (unit.get("flags") or {}).get("image_context_missing")
        ),
    }
    if diagnostic_focus_label_ids is None:
        output_instructions = """你只输出proposed和checks，不输出selected或KEEP/REJECT。最终取舍由程序执行，reason无权修改checks。

对每个proposed候选必须输出一组checks：
- literal_relation只能是EXACT、ANALOGY_ONLY、FALSE。
- scope_kind只能是GENERIC、RESTRICTED。
- scope_anchors是Label名称中的范围限定词；RESTRICTED时至少一个，GENERIC时必须为空。不得把通用机制当作范围限定词。
- object_match只能是EXACT、NOT_APPLICABLE、MISMATCH。
- RESTRICTED必须提供object_evidence；quote中必须直接出现scope_anchors中的对象限定，不能用相同机制代替。
- mechanism_relation只能是DIRECT、SHARED_ONLY、BACKGROUND。
- question_target用一个短句写当前设问要求学生最终判断、计算、解释或产出什么。
- label_target用一个短句写该Label本身要解决的任务，不得按当前题目改写Label目标。
- target_relation只能是EXACT、PREREQUISITE_ONLY、DIFFERENT。只有两者要求得出同一类最终结论才是EXACT；只是前置知识、方法相似或最终判断对象不同时，必须是PREREQUISITE_ONLY或DIFFERENT。
- dimension_match和necessary只能是布尔值。
- evidence是支持当前Label考点的原文。

evidence和object_evidence的source只能是parent_stem、stem、options、answer_text、analysis；quote不超过60字，必须逐字复制连续原文，不得改写、拼接、填空、推导或补写。

只输出一个JSON对象：
{
  "proposed": ["C01", "C05"],
  "checks": {
    "C01": {
      "literal_relation": "EXACT",
      "scope_kind": "GENERIC",
      "scope_anchors": [],
      "object_match": "NOT_APPLICABLE",
      "object_evidence": null,
      "mechanism_relation": "DIRECT",
      "question_target": "当前设问要求得出的最终结论",
      "label_target": "Label要解决的任务",
      "target_relation": "EXACT",
      "dimension_match": true,
      "necessary": true,
      "evidence": {"source": "stem", "quote": "题目连续原文"}
    },
    "C05": {
      "literal_relation": "EXACT",
      "scope_kind": "RESTRICTED",
      "scope_anchors": ["具体对象"],
      "object_match": "EXACT",
      "object_evidence": {"source": "analysis", "quote": "含具体对象的连续原文"},
      "mechanism_relation": "DIRECT",
      "question_target": "当前设问要求得出的最终结论",
      "label_target": "Label要解决的任务",
      "target_relation": "EXACT",
      "dimension_match": true,
      "necessary": true,
      "evidence": {"source": "analysis", "quote": "解析连续原文"}
    }
  },
  "context_insufficient": false,
  "need_expand_recall": false,
  "reason": "简要说明题目实际考查什么，不得用reason推翻checks"
}
不要输出Markdown或JSON之外的内容。"""
    else:
        focus_codes = [
            code
            for code, label_id in code_map.items()
            if label_id in diagnostic_focus_label_ids
        ]
        output_instructions = f"""这是单题诊断模式，不限制reason和detailed_reason为短句。
在正常判标后，必须详细审核：
1. 所有最终selected候选；
2. 指定的重点候选：{json.dumps(focus_codes, ensure_ascii=False)}。

对每个被审核候选分别回答：
- 完整label_name代入“本题直接考查【label_name】”是否字面成立；
- 题目对象与Label的物种、疾病、实验、材料、组织或场景限定是否一致；
- 是否只是共享底层机制、类比或同类实例；
- 不会该Label是否仍能完整解题；
- 支持选择的原文证据和反对选择的证据；
- 最终KEEP或REJECT。

evidence.quote可不超过300字，但仍必须是指定source字段的连续原文。
只输出一个JSON对象：
{{
  "selected": ["C01"],
  "evidence": {{
    "C01": {{"source": "analysis", "quote": "题目或解析的连续原文"}}
  }},
  "candidate_reviews": [
    {{
      "code": "C01",
      "decision": "KEEP",
      "label_name_literal_test": "通过或不通过，并解释",
      "object_scope_match": "对象范围是否一致",
      "shared_mechanism_only": false,
      "necessary_for_solution": true,
      "supporting_evidence": {{"source": "analysis", "quote": "连续原文"}},
      "counterevidence": "题目中反对该Label的对象、维度或边界证据",
      "detailed_reason": "详细说明为什么KEEP或REJECT"
    }}
  ],
  "context_insufficient": false,
  "need_expand_recall": false,
  "reason": "详细总结最终选择，特别说明重点候选为什么被选或被拒绝"
}}
不要输出Markdown或JSON之外的内容。"""
    prompt = f"""你是严谨的高中生物知识点候选审核器。本任务采用非对称损失：错标的代价远高于漏标。可以少提名、置空或扩召。你不拥有最终选择权；最终Label由程序根据结构化硬门槛计算。

任务流程（内部完成判断，只输出简短结论依据，不输出详细思考过程）：
A. 先用当前设问、选项判断、答案和解析列出“完成本题必须调用的知识”。
B. 对每个拟选Label逐项通过下面七道硬门槛。
C. 对暂定的proposed做一次反证复核：主动寻找“为什么它不该被提名”的证据。你如实填写checks，不得用reason改变门槛结果。

七道硬门槛（必须全部通过）：
1. 直接考查：正确解答当前设问确实需要该Label；仅出现于材料、父题背景、工具名或弱联想不通过。错误选项只有在判断其错误必须调用该知识时才算直接考查。
2. Label范围优先：Label允许覆盖的范围由label_name、label_path、definition和distinctions共同确定。core_concepts只能解释已由上述字段确定的范围，不能扩大范围。只命中core_concepts中一个方法、实例、名词或底层机制，不足以选择该Label。
3. 考查维度：原理、现象、实验操作、实验设计、应用、方法、结论、发展史是不同维度，不能互相替代。
4. 对象与限定词：只有名称和定义本身是通用原理、规律或机制的Label，才允许跨材料应用。如果Label名称或definition包含特定疾病、物种、实验、材料、组织或应用场景，当前题目、答案或解析必须出现相同对象或明确考查该对象；不得因底层机制相同而迁移具体Label。
5. 生命层级与作用通道：必须保持题目实际考查的对象、动作、生命层级、结构和作用通道一致。不得改写或补写题目中没有的对象、实验、结构、过程或作用通道。题目明确考查跨层级因果关系时才允许合理多标。
6. 边界否决：distinctions是硬否决条件；只要题目落在它排除的一侧，立即拒绝。实验/方法Label还必须真正考实验目的、步骤、变量、现象、误差或方案评价，不得由同模块概念触发。
7. 必要性反问：如果学生完全不会该Label，仍能依靠其他知识完整解决当前设问，则该Label不是必要考点，拒绝。

任务目标一致性（独立硬门槛）：
- 先写question_target：当前设问最终要求学生得出什么。
- 再仅根据label_name、label_path和definition写label_target：该Label要解决什么。
- 两者的研究对象、判断动作和最终结论都一致时，target_relation才是EXACT。
- 同样使用杂交、计数、实验或计算方法，但最终要判断的事物不同，必须是DIFFERENT。
- Label只是完成当前任务的背景或前置知识，必须是PREREQUISITE_ONLY。

Label名称字面成立测试（选择前必做）：
- 把完整label_name代入“本题直接考查【label_name】”。
- 只有这句话对当前题目字面成立，且不需要“类比、类似、共享机制、可迁移、属于同类”等转换才能选择。
- 题目只考通用机制而label_name指向特定疾病、物种、实验、材料或场景时，拒绝该具体Label，只能选通用Label。

选择规则：
8. 只判断当前小题。parent_stem仅补足语境；父题其他内容和兄弟小题不选。
9. 合理多标可以保留，但每一个Label都必须独立通过全部七道门槛；不得因已有一个正确Label就顺带加入相关Label。不得因为研究对象、题干关键词或所属章节相同，就用考查机制或维度不同的Label替代。
10. 每个proposed都必须提供一条可机器校验的evidence：source只能是parent_stem、stem、options、answer_text或analysis，quote必须是该字段中连续复制的简短原文。
11. 题目有明确生物考点，但没有任何候选值得提名时，proposed=[]且need_expand_recall=true。宁可置空，不得提名“最接近”的替代Label。
12. 若已有安全Label，但可能漏掉不确定次要项，不要用猜测补齐；保留安全Label即可。
13. 仅当缺图或缺父题材料导致连一个可靠Label都无法提名时，才设context_insufficient=true。因此proposed非空时context_insufficient必须为false。
14. 非生物题或无有效设问：proposed=[]，checks={{}}，need_expand_recall=false，context_insufficient=false。

只能返回C01等短代码，不能抄写长label_id。

题目：
{json.dumps(question, ensure_ascii=False)}

候选Label（顺序不代表最终正确性）：
{json.dumps(candidate_cards, ensure_ascii=False)}

{output_instructions}"""
    return prompt, code_map


def validate_adjudication_result(
    value: dict[str, Any],
    known_codes: set[str],
) -> dict[str, Any]:
    required = (
        "reason",
        "proposed",
        "checks",
        "need_expand_recall",
        "context_insufficient",
    )
    for field in required:
        if field not in value:
            raise ValueError(f"missing {field}")
    for field in (
        "need_expand_recall",
        "context_insufficient",
    ):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
    unknown_codes: list[str] = []

    def normalize_codes(field: str) -> list[str]:
        values = value[field]
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        seen: set[str] = set()
        normalized_values: list[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError(f"{field} items must be short codes")
            code = item.strip()
            if code not in known_codes:
                if code and code not in unknown_codes:
                    unknown_codes.append(code)
                continue
            if code not in seen:
                normalized_values.append(code)
                seen.add(code)
        return normalized_values

    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    reason = reason.strip()
    if len(reason) > 1000:
        raise ValueError("reason is too long")

    proposed = normalize_codes("proposed")
    raw_checks = value["checks"]
    if not isinstance(raw_checks, dict):
        raise ValueError("checks must be an object keyed by proposed code")

    def normalize_evidence(
        evidence: Any, *, code: str, field: str, required: bool
    ) -> dict[str, str] | None:
        if evidence is None and not required:
            return None
        if not isinstance(evidence, dict):
            raise ValueError(f"missing {field} object for {code}")
        source = evidence.get("source")
        quote = evidence.get("quote")
        if source not in EVIDENCE_SOURCES:
            raise ValueError(f"invalid {field} source for {code}")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"missing non-empty {field} quote for {code}")
        quote = quote.strip()
        if len(quote) > 300:
            raise ValueError(f"{field} for {code} is too long")
        return {"source": source, "quote": quote}

    normalized_checks: dict[str, dict[str, Any]] = {}
    selected: list[str] = []
    gate_rejected_codes: dict[str, list[str]] = {}
    for code in proposed:
        check = raw_checks.get(code)
        if not isinstance(check, dict):
            raise ValueError(f"missing checks object for {code}")
        literal_relation = check.get("literal_relation")
        scope_kind = check.get("scope_kind")
        object_match = check.get("object_match")
        mechanism_relation = check.get("mechanism_relation")
        target_relation = check.get("target_relation")
        if literal_relation not in {"EXACT", "ANALOGY_ONLY", "FALSE"}:
            raise ValueError(f"invalid literal_relation for {code}")
        if scope_kind not in {"GENERIC", "RESTRICTED"}:
            raise ValueError(f"invalid scope_kind for {code}")
        if object_match not in {"EXACT", "NOT_APPLICABLE", "MISMATCH"}:
            raise ValueError(f"invalid object_match for {code}")
        if mechanism_relation not in {"DIRECT", "SHARED_ONLY", "BACKGROUND"}:
            raise ValueError(f"invalid mechanism_relation for {code}")
        if target_relation not in {"EXACT", "PREREQUISITE_ONLY", "DIFFERENT"}:
            raise ValueError(f"invalid target_relation for {code}")
        question_target = check.get("question_target")
        label_target = check.get("label_target")
        for field, text in (
            ("question_target", question_target),
            ("label_target", label_target),
        ):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{field} must be a non-empty string for {code}")
            if len(text.strip()) > 300:
                raise ValueError(f"{field} is too long for {code}")
        if not isinstance(check.get("dimension_match"), bool):
            raise ValueError(f"dimension_match must be boolean for {code}")
        if not isinstance(check.get("necessary"), bool):
            raise ValueError(f"necessary must be boolean for {code}")
        anchors = check.get("scope_anchors")
        if not isinstance(anchors, list) or any(
            not isinstance(anchor, str) or not anchor.strip() for anchor in anchors
        ):
            raise ValueError(f"scope_anchors must be non-empty strings for {code}")
        anchors = list(dict.fromkeys(anchor.strip() for anchor in anchors))
        evidence = normalize_evidence(
            check.get("evidence"), code=code, field="evidence", required=True
        )
        object_evidence = normalize_evidence(
            check.get("object_evidence"),
            code=code,
            field="object_evidence",
            required=scope_kind == "RESTRICTED",
        )
        normalized_check = {
            "literal_relation": literal_relation,
            "scope_kind": scope_kind,
            "scope_anchors": anchors,
            "object_match": object_match,
            "object_evidence": object_evidence,
            "mechanism_relation": mechanism_relation,
            "question_target": question_target.strip(),
            "label_target": label_target.strip(),
            "target_relation": target_relation,
            "dimension_match": check["dimension_match"],
            "necessary": check["necessary"],
            "evidence": evidence,
        }
        normalized_checks[code] = normalized_check

        failures: list[str] = []
        if literal_relation != "EXACT":
            failures.append("literal_relation")
        if scope_kind == "GENERIC":
            if anchors:
                failures.append("generic_scope_has_anchors")
            if object_match != "NOT_APPLICABLE":
                failures.append("generic_object_match")
            if object_evidence is not None:
                failures.append("generic_object_evidence")
        else:
            if not anchors:
                failures.append("restricted_scope_missing_anchors")
            if object_match != "EXACT":
                failures.append("restricted_object_mismatch")
            if object_evidence is None:
                failures.append("restricted_object_evidence")
        if mechanism_relation != "DIRECT":
            failures.append("mechanism_relation")
        if target_relation != "EXACT":
            failures.append("target_relation")
        if not check["dimension_match"]:
            failures.append("dimension_match")
        if not check["necessary"]:
            failures.append("necessary")
        if failures:
            gate_rejected_codes[code] = failures
        else:
            selected.append(code)

    normalized_evidence = {
        code: normalized_checks[code]["evidence"] for code in selected
    }
    context_forced_false = bool(selected and value["context_insufficient"])
    return {
        "reason": reason,
        "proposed": proposed,
        "checks": normalized_checks,
        "selected": selected,
        "evidence": normalized_evidence,
        "gate_rejected_codes": gate_rejected_codes,
        "unknown_selected_codes_dropped": unknown_codes,
        "context_insufficient": False
        if selected
        else value["context_insufficient"],
        "context_insufficient_forced_false": context_forced_false,
        "need_expand_recall": value["need_expand_recall"] or bool(unknown_codes),
        "none_of_candidates": not bool(selected),
    }


def _latest_success(
    evidence_path: Path, *, prompt_version: str
) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    rows = 0
    if not evidence_path.exists():
        return latest, rows
    for record in _read_jsonl(evidence_path):
        rows += 1
        if (
            record.get("prompt_version") == prompt_version
            and not record.get("error")
            and isinstance(record.get("parsed_response"), dict)
        ):
            latest[str(record["question_id"])] = record
    return latest, rows


def run_adjudication(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    limit: int | None = None,
    question_ids: set[str] | None = None,
    max_tokens: int = 1024,
    workers: int = 1,
) -> dict[str, Any]:
    run_started = time.monotonic()
    run_started_at = datetime.now(timezone.utc).isoformat()
    if workers < 1:
        raise ValueError("workers must be positive")
    units = _read_jsonl(units_path)
    if question_ids is not None:
        units = [
            unit
            for unit in units
            if str(unit.get("question_id") or "") in question_ids
        ]
        found_ids = {str(unit.get("question_id") or "") for unit in units}
        missing_requested = sorted(question_ids - found_ids)
        if missing_requested:
            raise ValueError(
                f"question_id not found in units: {missing_requested[0]}"
            )
    if limit is not None:
        units = units[:limit]
    candidate_rows = {
        str(row["question_id"]): row for row in _read_jsonl(candidates_path)
    }
    labels_by_id = {
        str(row["label_id"]): row for row in _read_jsonl(labels_path)
    }
    expected_ids = [str(unit.get("question_id") or "") for unit in units]
    if any(not question_id for question_id in expected_ids):
        raise ValueError("every unit must have question_id")
    missing = [question_id for question_id in expected_ids if question_id not in candidate_rows]
    if missing:
        raise ValueError(f"missing candidates for question_id: {missing[0]}")

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.jsonl"
    prompt_version = PROMPT_VERSION
    candidate_versions = sorted(
        {str(row.get("retrieval_version") or "") for row in candidate_rows.values()}
    )
    candidate_count_distribution = dict(
        sorted(
            Counter(
                str(len(row.get("candidates") or []))
                for row in candidate_rows.values()
            ).items(),
            key=lambda item: int(item[0]),
        )
    )
    manifest = {
        "prompt_version": prompt_version,
        "model": model,
        "limit": limit,
        "question_ids": sorted(question_ids) if question_ids is not None else None,
        "max_tokens": max_tokens,
        "input_paths": {
            "units": str(Path(units_path)),
            "candidates": str(Path(candidates_path)),
            "labels": str(Path(labels_path)),
        },
        "input_sha256": {
            "units": _file_sha256(units_path),
            "candidates": _file_sha256(candidates_path),
            "labels": _file_sha256(labels_path),
        },
        "candidate_retrieval_versions": candidate_versions,
        "candidate_count_distribution": candidate_count_distribution,
    }
    _ensure_run_manifest(output_dir / "run_manifest.json", manifest)
    completed, evidence_rows = _latest_success(
        evidence_path, prompt_version=prompt_version
    )
    requests_succeeded = 0
    requests_failed = 0
    pending_units: list[tuple[int, dict[str, Any]]] = []
    for index, unit in enumerate(units, 1):
        question_id = str(unit["question_id"])
        if question_id in completed:
            continue
        candidates = candidate_rows[question_id].get("candidates") or []
        unknown = [
            str(candidate.get("label_id"))
            for candidate in candidates
            if str(candidate.get("label_id")) not in labels_by_id
        ]
        if unknown:
            raise ValueError(f"candidate uses unknown label_id: {unknown[0]}")
        pending_units.append((index, unit))

    def adjudicate(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, unit = item
        question_id = str(unit["question_id"])
        candidates = candidate_rows[question_id].get("candidates") or []
        prompt, code_map = build_adjudication_prompt(unit, candidates, labels_by_id)
        record = {
            "stage": "candidate_adjudication",
            "prompt_version": prompt_version,
            "question_id": question_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "candidate_code_map": code_map,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
            "usage": None,
            "reasoning": None,
            "response_message_keys": [],
            "retry_errors": [],
            "error": None,
        }
        try:
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": "你是严谨的高中生物知识点判标器，只输出JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            record.update(
                {
                    "raw_response": response.content,
                    "endpoint": response.endpoint,
                    "attempts": response.attempts,
                    "latency_seconds": response.latency_seconds,
                    "usage": getattr(response, "usage", None),
                    "reasoning": getattr(response, "reasoning", None),
                    "response_message_keys": list(
                        getattr(response, "response_message_keys", ())
                    ),
                    "retry_errors": list(getattr(response, "retry_errors", ())),
                }
            )
            record["parsed_response"] = validate_adjudication_result(
                parse_json_content(response.content),
                set(code_map),
            )
        except DSRequestError as exc:
            record.update(
                {
                    "endpoint": exc.endpoint,
                    "attempts": exc.attempts,
                    "latency_seconds": exc.latency_seconds,
                    "retry_errors": list(exc.retry_errors),
                }
            )
            record["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return index, record

    def persist(results: Any) -> None:
        nonlocal completed, evidence_rows, requests_succeeded, requests_failed
        finished_this_run = 0
        for original_index, record in results:
            finished_this_run += 1
            question_id = str(record["question_id"])
            requests_failed += int(bool(record["error"]))
            requests_succeeded += int(not record["error"])
            append_evidence(evidence_path, record)
            evidence_rows += 1
            if not record["error"]:
                completed[question_id] = record
            interim = {
                "input": len(units),
                "success": len(completed),
                "error": len(units) - len(completed),
                "pending": len(units) - len(completed),
                "evidence_rows": evidence_rows,
                "requests_succeeded_this_run": requests_succeeded,
                "requests_failed_this_run": requests_failed,
                "workers": workers,
                "model": model,
                "prompt_version": prompt_version,
            }
            _write_json_atomic(output_dir / "report.json", interim)
            print(
                f"[{finished_this_run}/{len(pending_units)}; source={original_index}/{len(units)}] "
                f"{question_id} {'ERROR' if record['error'] else 'OK'}",
                flush=True,
            )

    if workers == 1:
        persist(map(adjudicate, pending_units))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            persist(executor.map(adjudicate, pending_units))

    completed, evidence_rows = _latest_success(
        evidence_path, prompt_version=prompt_version
    )
    predictions_path = output_dir / "predictions.jsonl"
    temporary = predictions_path.with_name(f".{predictions_path.name}.tmp")
    tail_path = output_dir / "tail_selected.jsonl"
    tail_temporary = tail_path.with_name(f".{tail_path.name}.tmp")
    selected_count_distribution: Counter[str] = Counter()
    selected_rank_distribution: Counter[str] = Counter()
    selected_from_tail = 0
    questions_using_tail = 0
    selected_from_rank_21_plus = 0
    questions_using_rank_21_plus = 0
    selected_from_rank_26_30 = 0
    questions_using_rank_26_30 = 0
    max_selected_rank = 0
    need_expand = 0
    none_count = 0
    context_insufficient_count = 0
    context_insufficient_forced_false_count = 0
    unverified_evidence_items = 0
    questions_with_unverified_evidence = 0
    gate_rejected_label_count = 0
    questions_with_gate_rejections = 0
    gate_rejection_reasons: Counter[str] = Counter()
    unknown_selected_codes_dropped_count = 0
    questions_with_unknown_selected_codes = 0
    usable_for_training_count = 0
    training_filter_reasons: Counter[str] = Counter()
    with (
        temporary.open("w", encoding="utf-8", newline="\n") as output,
        tail_temporary.open("w", encoding="utf-8", newline="\n") as tail_output,
    ):
        for unit in units:
            question_id = str(unit["question_id"])
            record = completed.get(question_id)
            if not record:
                continue
            parsed = record["parsed_response"]
            code_map = record["candidate_code_map"]
            candidates = candidate_rows[question_id].get("candidates") or []
            candidates_by_id = {
                str(candidate["label_id"]): candidate for candidate in candidates
            }
            selected_labels = []
            used_tail = False
            used_rank_21_plus = False
            used_rank_26_30 = False
            used_unverified_evidence = False
            gate_rejected_labels = []
            for code, reasons in parsed.get("gate_rejected_codes", {}).items():
                label_id = code_map[code]
                label = labels_by_id[label_id]
                gate_rejected_labels.append(
                    {
                        "label_id": label_id,
                        "label_name": label.get("label_name", ""),
                        "code": code,
                        "reasons": reasons,
                        "stage": "structured_checks",
                        "checks": parsed["checks"].get(code, {}),
                    }
                )
                gate_rejected_label_count += 1
                gate_rejection_reasons.update(reasons)
            for code in parsed["selected"]:
                label_id = code_map[code]
                label = labels_by_id[label_id]
                candidate = candidates_by_id[label_id]
                check = parsed["checks"][code]
                evidence = parsed["evidence"][code]
                evidence_source = evidence["source"]
                evidence_quote = evidence["quote"]
                source_text = str(unit.get(evidence_source) or "")
                evidence_verified = bool(_normalize_evidence_text(evidence_quote)) and (
                    _normalize_evidence_text(evidence_quote)
                    in _normalize_evidence_text(source_text)
                )
                verification_failures = []
                if not evidence_verified:
                    verification_failures.append("evidence_not_verbatim")
                object_evidence = check.get("object_evidence")
                object_evidence_verified = True
                scope_anchors_verified = True
                if check["scope_kind"] == "RESTRICTED":
                    normalized_label_name = _normalize_evidence_text(
                        label.get("label_name", "")
                    )
                    normalized_anchors = [
                        _normalize_evidence_text(anchor)
                        for anchor in check["scope_anchors"]
                    ]
                    scope_anchors_verified = bool(normalized_anchors) and all(
                        len(anchor) >= 2 and anchor in normalized_label_name
                        for anchor in normalized_anchors
                    )
                    if not scope_anchors_verified:
                        verification_failures.append("scope_anchor_not_in_label_name")
                    object_source = object_evidence["source"]
                    object_quote = object_evidence["quote"]
                    object_source_text = str(unit.get(object_source) or "")
                    normalized_object_quote = _normalize_evidence_text(object_quote)
                    object_evidence_verified = bool(normalized_object_quote) and (
                        normalized_object_quote
                        in _normalize_evidence_text(object_source_text)
                    ) and any(
                        anchor in normalized_object_quote
                        for anchor in normalized_anchors
                    )
                    if not object_evidence_verified:
                        verification_failures.append("object_evidence_scope_mismatch")
                used_unverified_evidence = (
                    used_unverified_evidence or bool(verification_failures)
                )
                unverified_evidence_items += len(verification_failures)
                if verification_failures:
                    gate_rejected_labels.append(
                        {
                            "label_id": label_id,
                            "label_name": label.get("label_name", ""),
                            "code": code,
                            "reasons": verification_failures,
                            "stage": "evidence_verification",
                            "checks": check,
                        }
                    )
                    gate_rejected_label_count += 1
                    gate_rejection_reasons.update(verification_failures)
                    continue
                rank = int(candidate.get("candidate_rank") or candidate.get("rank") or 0)
                if rank > 0:
                    selected_rank_distribution[str(rank)] += 1
                    max_selected_rank = max(max_selected_rank, rank)
                used_tail = used_tail or 21 <= rank <= 25
                selected_from_tail += int(21 <= rank <= 25)
                used_rank_21_plus = used_rank_21_plus or rank >= 21
                selected_from_rank_21_plus += int(rank >= 21)
                used_rank_26_30 = used_rank_26_30 or 26 <= rank <= 30
                selected_from_rank_26_30 += int(26 <= rank <= 30)
                selected_labels.append(
                    {
                        "label_id": label_id,
                        "label_name": label.get("label_name", ""),
                        "label_path": format_label_path(label.get("label_path")),
                        "candidate_rank": rank,
                        "sources": candidate.get("sources", []),
                        "sparse_rank": candidate.get("sparse_rank"),
                        "dense_rank": candidate.get("dense_rank"),
                        "evidence_source": evidence_source,
                        "evidence": evidence_quote,
                        "evidence_verified": evidence_verified,
                        "scope_kind": check["scope_kind"],
                        "scope_anchors": check["scope_anchors"],
                        "literal_relation": check["literal_relation"],
                        "mechanism_relation": check["mechanism_relation"],
                        "question_target": check["question_target"],
                        "label_target": check["label_target"],
                        "target_relation": check["target_relation"],
                        "dimension_match": check["dimension_match"],
                        "necessary": check["necessary"],
                        "object_evidence": object_evidence,
                        "object_evidence_verified": object_evidence_verified,
                    }
                )
            selected_labels.sort(
                key=lambda item: (item["candidate_rank"], item["label_id"])
            )
            questions_using_tail += int(used_tail)
            questions_using_rank_21_plus += int(used_rank_21_plus)
            questions_using_rank_26_30 += int(used_rank_26_30)
            questions_with_unverified_evidence += int(used_unverified_evidence)
            questions_with_gate_rejections += int(bool(gate_rejected_labels))
            selected_count_distribution[str(len(selected_labels))] += 1
            need_expand += int(parsed["need_expand_recall"])
            none_count += int(not selected_labels)
            context_insufficient_count += int(parsed["context_insufficient"])
            context_insufficient_forced_false_count += int(
                parsed.get("context_insufficient_forced_false", False)
            )
            dropped_unknown_codes = parsed.get("unknown_selected_codes_dropped", [])
            unknown_selected_codes_dropped_count += len(dropped_unknown_codes)
            questions_with_unknown_selected_codes += int(bool(dropped_unknown_codes))
            text_content_missing = not str(unit.get("stem") or "").strip() and not str(
                unit.get("parent_stem") or ""
            ).strip()
            needs_review = bool(
                parsed["need_expand_recall"]
                or parsed["context_insufficient"]
                or text_content_missing
                or used_unverified_evidence
            )
            usable_for_training = bool(
                selected_labels
                and not parsed["need_expand_recall"]
                and not parsed["context_insufficient"]
                and not text_content_missing
                and not used_unverified_evidence
            )
            usable_for_training_count += int(usable_for_training)
            if not selected_labels:
                training_filter_reasons["empty_selected"] += 1
            if parsed["need_expand_recall"]:
                training_filter_reasons["need_expand_recall"] += 1
            if parsed["context_insufficient"]:
                training_filter_reasons["context_insufficient"] += 1
            if text_content_missing:
                training_filter_reasons["missing_question_text"] += 1
            if used_unverified_evidence:
                training_filter_reasons["unverified_evidence"] += 1
            prediction = {
                "question_id": question_id,
                "parent_id": unit.get("parent_id", question_id),
                "unit_type": unit.get("unit_type", ""),
                "reason": parsed["reason"],
                "selected_labels": selected_labels,
                "proposed_codes": parsed.get("proposed", []),
                "gate_rejected_labels": gate_rejected_labels,
                "unknown_selected_codes_dropped": parsed.get(
                    "unknown_selected_codes_dropped", []
                ),
                "none_of_candidates": not bool(selected_labels),
                "need_expand_recall": parsed["need_expand_recall"],
                "context_insufficient": parsed["context_insufficient"],
                "text_content_missing": text_content_missing,
                "needs_review": needs_review,
                "usable_for_training": usable_for_training,
                "candidate_count": len(candidates),
                "retrieval_version": candidate_rows[question_id].get(
                    "retrieval_version", ""
                ),
                "model": model,
                "prompt_version": prompt_version,
            }
            output.write(
                json.dumps(prediction, ensure_ascii=False, sort_keys=True)
            )
            output.write("\n")
            if used_rank_21_plus:
                tail_output.write(
                    json.dumps(
                        {
                            "question": {
                                "question_id": question_id,
                                "parent_stem": unit.get("parent_stem", ""),
                                "stem": unit.get("stem", ""),
                                "options": unit.get("options", ""),
                                "answer_text": unit.get("answer_text", ""),
                                "analysis": unit.get("analysis", ""),
                            },
                            "prediction": prediction,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                tail_output.write("\n")
    temporary.replace(predictions_path)
    tail_temporary.replace(tail_path)
    success = len(completed)
    latencies = sorted(
        float(record["latency_seconds"])
        for record in completed.values()
        if isinstance(record.get("latency_seconds"), (int, float))
    )

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = max(0, math.ceil(len(values) * fraction) - 1)
        return round(values[index], 3)

    run_wall_seconds = round(time.monotonic() - run_started, 3)
    requests_this_run = requests_succeeded + requests_failed
    usages = [
        record["usage"]
        for record in completed.values()
        if isinstance(record.get("usage"), dict)
    ]
    prompt_tokens = [
        int(usage["prompt_tokens"])
        for usage in usages
        if isinstance(usage.get("prompt_tokens"), (int, float))
    ]
    completion_tokens = [
        int(usage["completion_tokens"])
        for usage in usages
        if isinstance(usage.get("completion_tokens"), (int, float))
    ]
    total_tokens = [
        int(usage["total_tokens"])
        for usage in usages
        if isinstance(usage.get("total_tokens"), (int, float))
    ]
    prompt_chars = [
        int(record["prompt_chars"])
        for record in completed.values()
        if isinstance(record.get("prompt_chars"), int)
    ]
    reasoning_requests = sum(
        bool(record.get("reasoning")) for record in completed.values()
    )
    retry_error_types: Counter[str] = Counter()
    requests_retried = 0
    for record in completed.values():
        retry_errors = record.get("retry_errors") or []
        requests_retried += int(bool(retry_errors))
        retry_error_types.update(
            str(error.get("error_type") or "unknown")
            for error in retry_errors
            if isinstance(error, dict)
        )
    report = {
        "input": len(units),
        "processed": success,
        "success": success,
        "error": len(units) - success,
        "pending": len(units) - success,
        "evidence_rows": evidence_rows,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "requests_retried": requests_retried,
        "retry_error_types": dict(sorted(retry_error_types.items())),
        "workers": workers,
        "run_started_at": run_started_at,
        "run_wall_seconds": run_wall_seconds,
        "requests_per_second_this_run": round(
            requests_this_run / run_wall_seconds, 4
        )
        if run_wall_seconds
        else None,
        "request_latency_seconds": {
            "count": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": round(latencies[-1], 3) if latencies else None,
        },
        "token_usage": {
            "requests_with_usage": len(usages),
            "total_prompt_tokens": sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
            "total_tokens": sum(total_tokens),
            "mean_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3)
            if prompt_tokens
            else None,
            "mean_completion_tokens": round(
                sum(completion_tokens) / len(completion_tokens), 3
            )
            if completion_tokens
            else None,
            "mean_total_tokens": round(sum(total_tokens) / len(total_tokens), 3)
            if total_tokens
            else None,
            "mean_prompt_chars": round(sum(prompt_chars) / len(prompt_chars), 3)
            if prompt_chars
            else None,
            "requests_with_reasoning": reasoning_requests,
        },
        "selected_count_distribution": dict(
            sorted(selected_count_distribution.items(), key=lambda item: int(item[0]))
        ),
        "selected_candidate_rank_distribution": dict(
            sorted(selected_rank_distribution.items(), key=lambda item: int(item[0]))
        ),
        "max_selected_candidate_rank": max_selected_rank,
        "selected_from_rank_21_25": selected_from_tail,
        "questions_using_rank_21_25": questions_using_tail,
        "selected_from_rank_21_plus": selected_from_rank_21_plus,
        "questions_using_rank_21_plus": questions_using_rank_21_plus,
        "selected_from_rank_26_30": selected_from_rank_26_30,
        "questions_using_rank_26_30": questions_using_rank_26_30,
        "need_expand_recall": need_expand,
        "none_of_candidates": none_count,
        "context_insufficient": context_insufficient_count,
        "context_insufficient_forced_false": context_insufficient_forced_false_count,
        "unverified_evidence_items": unverified_evidence_items,
        "questions_with_unverified_evidence": questions_with_unverified_evidence,
        "gate_rejected_label_count": gate_rejected_label_count,
        "questions_with_gate_rejections": questions_with_gate_rejections,
        "gate_rejection_reasons": dict(sorted(gate_rejection_reasons.items())),
        "unknown_selected_codes_dropped": unknown_selected_codes_dropped_count,
        "questions_with_unknown_selected_codes": questions_with_unknown_selected_codes,
        "usable_for_training": usable_for_training_count,
        "filtered_from_training": success - usable_for_training_count,
        "training_filter_reasons": dict(sorted(training_filter_reasons.items())),
        "input_sha256": manifest["input_sha256"],
        "candidate_retrieval_versions": candidate_versions,
        "candidate_count_distribution": candidate_count_distribution,
        "model": model,
        "prompt_version": prompt_version,
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
