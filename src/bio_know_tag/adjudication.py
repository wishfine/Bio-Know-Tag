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


PROMPT_VERSION = "candidate-adjudication-v8.6-v83-reason-last"
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
                "common_assessments": label.get("common_assessments", ""),
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
    prompt = f"""你是严谨的高中生物知识点判标器。本任务采用非对称损失：错标的代价远高于漏标。可以少选、置空或扩召，绝不得把只是相关、更宽泛或边界不同的Label写入selected。

任务流程（内部完成判断，只输出简短结论依据，不输出详细思考过程）：
A. 先用当前设问、选项判断、答案和解析列出“完成本题必须调用的知识”。
B. 对每个拟选Label逐项通过下面五道硬门槛。
C. 对暂定的selected做一次反证复核：主动寻找“为什么它不该被选”的证据。只要任意一道不能确定通过，就从selected删除。

五道硬门槛（必须全部通过）：
1. 直接考查：正确解答当前设问确实需要该Label；仅出现于材料、父题背景、工具名或弱联想不通过。错误选项只有在判断其错误必须调用该知识时才算直接考查。
2. 精确定义：题目考点完整落入definition和core_concepts，不得因共享一个名词就扩张Label。
3. 考查维度：原理、现象、实验操作、实验设计、应用、方法、结论、发展史是不同维度，不能互相替代。common_assessments可帮助判定维度，但不能单独证明应入选。
4. 边界否决：distinctions是硬否决条件；只要题目落在它排除的一侧，立即拒绝。
5. 必要性反问：如果学生完全不会该Label，仍能依靠其他知识完整解决当前设问，则该Label不是必要考点，拒绝。

选择规则：
6. 只判断当前小题。parent_stem仅补足语境；父题其他内容和兄弟小题不选。
7. 合理多标可以保留，但每一个Label都必须独立通过全部五道门槛；不得因已有一个正确Label就顺带加入相关Label。不得因为研究对象、题干关键词或所属章节相同，就用考查机制或维度不同的Label替代。
8. 反证复核失败的候选直接从selected删除；reason只概括最终选择或置空的依据，不逐项输出被拒绝候选或详细思考过程。
9. 题目有明确生物考点，但没有任何候选能通过五道门槛时，selected=[]且need_expand_recall=true。宁可置空，不得选“最接近”的替代Label。
10. 若已有安全Label，但可能漏掉不确定次要项，不要用猜测补齐；保留安全Label即可。
11. 仅当缺图或缺父题材料导致连一个可靠Label都无法确定时，才设context_insufficient=true。答案或解析足以判断时必须为false。
12. 非生物题或无有效设问：selected=[]，need_expand_recall=false，context_insufficient=false。

只能返回C01等短代码，不能抄写长label_id。

题目：
{json.dumps(question, ensure_ascii=False)}

候选Label（顺序不代表最终正确性）：
{json.dumps(candidate_cards, ensure_ascii=False)}

reason用1至2句话、不超过120字，说明当前设问、答案或解析如何支持selected；若置空，则说明是候选不匹配、需要扩召还是上下文不足。不要罗列全部候选，不要输出详细思考过程。

只输出一个JSON对象：
{{
  "selected": ["C01", "C05"],
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
        "need_expand_recall",
        "context_insufficient",
    )
    for field in required:
        if field not in value:
            raise ValueError(f"missing {field}")
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
                raise ValueError(f"unknown {field} code: {code}")
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
    for field in (
        "need_expand_recall",
        "context_insufficient",
    ):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
    return {
        "reason": reason,
        "selected": normalized,
        "context_insufficient": value["context_insufficient"],
        "need_expand_recall": value["need_expand_recall"],
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
    max_selected_rank = 0
    need_expand = 0
    none_count = 0
    context_insufficient_count = 0
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
                selected_labels.append(
                    {
                        "label_id": label_id,
                        "label_name": label.get("label_name", ""),
                        "label_path": format_label_path(label.get("label_path")),
                        "candidate_rank": rank,
                        "sources": candidate.get("sources", []),
                        "sparse_rank": candidate.get("sparse_rank"),
                        "dense_rank": candidate.get("dense_rank"),
                    }
                )
            selected_labels.sort(
                key=lambda item: (item["candidate_rank"], item["label_id"])
            )
            questions_using_tail += int(used_tail)
            selected_count_distribution[str(len(selected_labels))] += 1
            need_expand += int(parsed["need_expand_recall"])
            none_count += int(parsed["none_of_candidates"])
            context_insufficient_count += int(parsed["context_insufficient"])
            needs_review = bool(
                parsed["need_expand_recall"]
                or parsed["context_insufficient"]
            )
            usable_for_training = bool(
                selected_labels
                and not parsed["need_expand_recall"]
                and not parsed["context_insufficient"]
            )
            usable_for_training_count += int(usable_for_training)
            if not selected_labels:
                training_filter_reasons["empty_selected"] += 1
            if parsed["need_expand_recall"]:
                training_filter_reasons["need_expand_recall"] += 1
            if parsed["context_insufficient"]:
                training_filter_reasons["context_insufficient"] += 1
            prediction = {
                "question_id": question_id,
                "parent_id": unit.get("parent_id", question_id),
                "unit_type": unit.get("unit_type", ""),
                "reason": parsed["reason"],
                "selected_labels": selected_labels,
                "none_of_candidates": parsed["none_of_candidates"],
                "need_expand_recall": parsed["need_expand_recall"],
                "context_insufficient": parsed["context_insufficient"],
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
            if used_tail:
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
        "need_expand_recall": need_expand,
        "none_of_candidates": none_count,
        "context_insufficient": context_insufficient_count,
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
