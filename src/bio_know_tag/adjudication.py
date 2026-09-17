"""DS adjudication over a small hybrid candidate set."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.ds import DSRequestError, append_evidence, parse_json_content
from bio_know_tag.retrieval import format_label_path


PROMPT_VERSION = "candidate-adjudication-v9.1d-method-purpose-alignment"
CANDIDATE_ORDER_VERSION = "candidate-adjudication-v8.3-internal-reflection"


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
    prompt = f"""你是严谨的高中生物知识点判标器。本任务高精度优先：错标的代价远高于漏标。可以少选、selected=[]或要求扩召；不得为提高覆盖率加入只是相关、同章节、上下位邻近、共享机制或常见伴随出现的Label。

任务是判断：当前小题是否直接考查候选Label所定义的知识范围，而不是寻找所有相关知识。只输出简短结论，不输出详细思考过程。

一、先界定Label
Label有效范围由label_name、label_path、definition和distinctions共同确定。distinctions是硬否决边界。core_concepts只用于解释该范围内的概念和机制，不能扩大Label范围；只命中其中一个方法、实例、名词或底层机制，不足以选中Label。

二、硬否决：任意一项成立就拒绝，后续不得翻回
1. 对象不一致：若Label被限定到特定物种、疾病、性状、实验、材料、组织、器官、细胞类型、技术或应用场景，则当前题目必须实际考查同一对象。不得因遗传方式、分子/生理机制、实验原理相同，而把具体对象A横向迁移到具体对象B的Label。具体对象题可以选其真正考查的通用上位机制Label。对象一致只是必要条件，不是选中条件。
2. 任务或维度不一致：原理/规律、结构、功能、现象、生理过程、实验原理、实验操作/设计/结果、判定方法、应用、结论和科学史不能互相替代。题目只是使用已知结论完成推断，不等于考查该结论的判定方法或发现实验。
3. 生命层级、结构或作用通道不一致：不得因宏观过程包含某个微观机制，或微观机制相似，就用不同层级的Label替代当前考点。
4. 当前小题范围不一致：只判断当前小题。parent_stem只能在当前小题存在“该患者、该实验、图中”等明确指代时补足对象和语境，不能单独制造考点。当前小题与父题背景的考查方向不同或冲突时，必须以当前小题的设问、答案和解析为唯一判标依据；parent_stem不得覆盖、扩张或替代当前设问的考点。父题其他内容和兄弟小题的知识不选。
5. 与distinctions冲突：若题目落在distinctions排除的一侧，立即拒绝。
6. 方法目的或结果指标不一致：对调查、取样、计数、测定、检测等Label，必须同时核对“对象是什么、要得到什么指标或结论、使用什么方法”。只有统计、计数、取样等动作词相同，或都涉及个体数量，但调查对象、目标指标或结果含义不同，必须拒绝。

三、还原当前任务
通过硬否决后，仅根据当前stem、options、answer_text和analysis，判断学生为了得出正确答案必须完成哪些具体判断。材料中出现的概念、解析为讲解完整而补充的背景，不自动算考点。错误选项只有在判断它错误必须调用该知识，且它构成题目的实质性考查而非孤立干扰信息时，才可支持该Label。

四、正向选中：必须同时满足
1. 范围命中：当前认知任务本身落在Label有效范围内，不是同章节、关键词相同、上下位相关、共享机制或类比实例。
2. 直接考查：该Label必须直接支持当前答案中一个关键判断。仅作为背景、材料对象、深层解释或“知道后理解更完整”不算直接考查。
3. 独立作用：多Label时，每个Label都必须独立解释当前题目中一个真实存在的判断任务。不得因一个Label成立就顺带加入父级、子级、同机制、同章节或常见搭配Label。上下位Label只有在题目分别直接考查两者时才可同选。
4. 关键判断：必须能指出“该Label直接支持了答案中哪一个关键判断”。如果只能说有帮助、相关或让理解更完整，则拒绝。不要因存在另一条解题路径，就拒绝一个本身被题目直接考查的合理Label。

五、特殊Label
1. 通用机制Label：若Label定义的是通用原理、规律、分类或方法，且题目确实直接应用它完成判断，可以跨不同材料实例选择。但具体对象A只能上溯到通用机制Label，不能横向迁移到共享机制的具体对象B Label。
2. 实验、方法、观察、调查、测定、制作、构建、判定类Label：只有当前设问真正要求学生判断对应目的、原理、步骤、变量、现象、结果、误差、方案或判定方法本身时才选择。选中前必须确认该Label的对象、方法目的和结果指标均与当前任务一致。只是使用该实验的结论、出现名称/材料，或利用已知对象信息做其他推断，都不选该类Label。
3. 综合Label：只有当前设问要求联动多个子知识得出一个联合结论时才选择。题目包含多个彼此独立的子知识，不等于考查综合Label；“综合、其他、应用”不得作为候选不精确时的兜底。

六、evidence与最终复核
每个selected Label必须提供一条不超过60字的evidence，从当前stem、options、answer_text或analysis中复制，不得推理补写。当前小题有明确指代时，parent_stem可用于证明对象，但不能单独证明考点。evidence必须支持该Label的直接考查，不是只证明二者相关。
生成selected前，对每个暂定Label反证复核：若它实际只是共享机制、同章节/上下位相关、另一具体对象、不同考查维度、背景补充或孤立干扰项，就删除。宁可少选，不做弱关联补标。

七、状态判断
轻微错别字/OCR异常若可由答案、解析和其他信息唯一消除，context_insufficient=false。只有缺图、缺父题或信息冲突导致连一个可靠Label都无法确定时，才设context_insufficient=true。
若当前小题有明确生物考点，但所有候选都无法成立：selected=[]、need_expand_recall=true。若已有可靠Label，只是怀疑存在不确定次要Label，保留可靠结果且need_expand_recall=false，不猜测补齐。非有效高中生物考查：selected=[]、need_expand_recall=false、context_insufficient=false。

只能返回C01等短代码，不能抄写长label_id。

题目：
{json.dumps(question, ensure_ascii=False)}

候选Label（顺序不代表最终正确性）：
{json.dumps(candidate_cards, ensure_ascii=False)}

候选顺序不代表正确性。多Label按候选出现顺序输出以减少波动。reason只说明最终选中或置空的核心原因，1至2句话、不超过120字，不输出详细分析。

只输出一个JSON对象：
{{
  "selected": ["C01", "C05"],
  "evidence": {{"C01": "题目原文", "C05": "题目原文"}},
  "context_insufficient": false,
  "need_expand_recall": false,
  "reason": "当前设问直接考查……"
}}
不要输出Markdown或JSON之外的内容。"""
    return prompt, code_map


def validate_adjudication_result(
    value: dict[str, Any],
    known_codes: set[str],
) -> dict[str, Any]:
    required = (
        "reason",
        "selected",
        "evidence",
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
    normalized = normalize_codes("selected")
    raw_evidence = value["evidence"]
    if not isinstance(raw_evidence, dict):
        raise ValueError("evidence must be an object keyed by selected code")
    normalized_evidence: dict[str, str] = {}
    for code in normalized:
        evidence = raw_evidence.get(code)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"missing non-empty evidence for {code}")
        evidence = evidence.strip()
        if len(evidence) > 300:
            raise ValueError(f"evidence for {code} is too long")
        normalized_evidence[code] = evidence
    return {
        "reason": reason,
        "selected": normalized,
        "evidence": normalized_evidence,
        "unknown_selected_codes_dropped": unknown_codes,
        "context_insufficient": value["context_insufficient"],
        "need_expand_recall": value["need_expand_recall"] or bool(unknown_codes),
        "none_of_candidates": not bool(normalized),
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
    max_tokens: int = 1024,
    workers: int = 1,
) -> dict[str, Any]:
    run_started = time.monotonic()
    run_started_at = datetime.now(timezone.utc).isoformat()
    if workers < 1:
        raise ValueError("workers must be positive")
    units = _read_jsonl(units_path)
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
            for code in parsed["selected"]:
                label_id = code_map[code]
                label = labels_by_id[label_id]
                candidate = candidates_by_id[label_id]
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
                        "evidence": parsed["evidence"][code],
                    }
                )
            selected_labels.sort(
                key=lambda item: (item["candidate_rank"], item["label_id"])
            )
            questions_using_tail += int(used_tail)
            questions_using_rank_21_plus += int(used_rank_21_plus)
            questions_using_rank_26_30 += int(used_rank_26_30)
            selected_count_distribution[str(len(selected_labels))] += 1
            need_expand += int(parsed["need_expand_recall"])
            none_count += int(parsed["none_of_candidates"])
            context_insufficient_count += int(parsed["context_insufficient"])
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
            )
            usable_for_training = bool(
                selected_labels
                and not parsed["need_expand_recall"]
                and not parsed["context_insufficient"]
                and not text_content_missing
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
            prediction = {
                "question_id": question_id,
                "parent_id": unit.get("parent_id", question_id),
                "unit_type": unit.get("unit_type", ""),
                "reason": parsed["reason"],
                "selected_labels": selected_labels,
                "unknown_selected_codes_dropped": parsed.get(
                    "unknown_selected_codes_dropped", []
                ),
                "none_of_candidates": parsed["none_of_candidates"],
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
