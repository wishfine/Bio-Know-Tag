"""DS adjudication over a small hybrid candidate set."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.ds import append_evidence, parse_json_content
from bio_know_tag.retrieval import format_label_path


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


def build_adjudication_prompt(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    code_map = {
        f"C{index:02d}": str(candidate["label_id"])
        for index, candidate in enumerate(candidates, 1)
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
        "question_id": str(unit.get("question_id") or ""),
        "unit_type": unit.get("unit_type", ""),
        "parent_stem": str(unit.get("parent_stem") or "")[:3000],
        "stem": str(unit.get("stem") or "")[:5000],
        "options": str(unit.get("options") or "")[:3000],
        "answer_text": str(unit.get("answer_text") or "")[:2000],
        "analysis": str(unit.get("analysis") or "")[:6000],
    }
    prompt = f"""你是严谨的高中生物知识点判标器。请从候选中选择完成当前设问所必需的最小充分知识点集合。

判标规则：
1. 依据老师给出的定义、核心概念和易混淆边界；不要参考旧knw_ids。
2. 只有正确解答当前设问需要调用的知识点才打；材料背景、实验工具、错误选项和干扰项不打。
3. 逐个设问、填空或正确选项判断知识需求；一个Label可以覆盖时，不再追加其上位概念、背景知识或底层常识。
4. 对每个准备选中的Label执行“删除测试”：删掉它以后，学生是否仍能仅凭其他已选Label完整回答对应设问？如果能，就不要选它。
5. 不能因为答案中的某个名词属于一个大类就给大类打标。例如题目只问膜的基本骨架时，不能仅因答案是“磷脂”而追加脂质分类；普通的引入天敌或生物农药也不等于利用生态系统信息传递。
6. 综合Label只有在题目确实要求多个子模块联动时才选。
7. 可以选择多个候选，也可以一个都不选。若正确知识点可能未进入候选，设置need_expand_recall=true。
8. 每个选中项的question_evidence必须逐字摘自本题的parent_stem、stem、options、answer_text或analysis，不能把Label释义、常识或推断伪装成题目证据；necessity说明该Label具体覆盖哪个设问。
9. 只能返回C01等短代码，不能抄写长label_id。

题目：
{json.dumps(question, ensure_ascii=False)}

候选Label（顺序不代表最终正确性）：
{json.dumps(candidate_cards, ensure_ascii=False)}

只输出一个JSON对象：
{{
  "selected": [{{"code": "C01", "question_evidence": "题目中的原文短句", "necessity": "该Label具体覆盖的设问及不可删除原因"}}],
  "rejected_close_codes": ["容易混淆但不应命中的候选代码"],
  "none_of_candidates": false,
  "need_expand_recall": false,
  "reason": "简洁说明最小充分集合的选择依据"
}}
不要输出Markdown或JSON之外的内容。"""
    return prompt, code_map


def validate_adjudication_result(
    value: dict[str, Any],
    known_codes: set[str],
    *,
    question_evidence_text: str | None = None,
) -> dict[str, Any]:
    required = (
        "selected",
        "rejected_close_codes",
        "none_of_candidates",
        "need_expand_recall",
        "reason",
    )
    for field in required:
        if field not in value:
            raise ValueError(f"missing {field}")
    selected = value["selected"]
    if not isinstance(selected, list):
        raise ValueError("selected must be a list")
    seen: set[str] = set()
    normalized = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("selected items must be objects")
        code = str(item.get("code") or "")
        evidence = str(item.get("question_evidence") or "").strip()
        necessity = str(item.get("necessity") or "").strip()
        if code not in known_codes:
            raise ValueError(f"unknown selected code: {code}")
        if not evidence:
            raise ValueError("selected question_evidence must be non-empty")
        if not necessity:
            raise ValueError("selected necessity must be non-empty")
        evidence_verified: bool | None = None
        if question_evidence_text is not None:
            normalized_source = re.sub(r"\s+", "", question_evidence_text)
            normalized_evidence = re.sub(r"\s+", "", evidence)
            evidence_verified = normalized_evidence in normalized_source
        if code not in seen:
            normalized.append(
                {
                    "code": code,
                    "question_evidence": evidence,
                    "necessity": necessity,
                    "question_evidence_verified": evidence_verified,
                }
            )
            seen.add(code)
    value["selected"] = normalized
    rejected = value["rejected_close_codes"]
    if not isinstance(rejected, list) or any(
        not isinstance(code, str) or code not in known_codes for code in rejected
    ):
        raise ValueError("rejected_close_codes contains an unknown code")
    value["rejected_close_codes"] = [
        code for code in dict.fromkeys(rejected) if code not in seen
    ]
    for field in ("none_of_candidates", "need_expand_recall"):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
    if bool(normalized) == value["none_of_candidates"]:
        raise ValueError("none_of_candidates is inconsistent with selected")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    return value


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
    prompt_version = "candidate-adjudication-v2"
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
            "candidate_code_map": code_map,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
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
                }
            )
            record["parsed_response"] = validate_adjudication_result(
                parse_json_content(response.content),
                set(code_map),
                question_evidence_text="\n".join(
                    str(unit.get(field) or "")
                    for field in (
                        "parent_stem",
                        "stem",
                        "options",
                        "answer_text",
                        "analysis",
                    )
                ),
            )
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
    unverified_evidence_items = 0
    questions_with_unverified_evidence = 0
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
            has_unverified_evidence = False
            for item in parsed["selected"]:
                label_id = code_map[item["code"]]
                label = labels_by_id[label_id]
                candidate = candidates_by_id[label_id]
                rank = int(candidate.get("candidate_rank") or candidate.get("rank") or 0)
                if rank > 0:
                    selected_rank_distribution[str(rank)] += 1
                    max_selected_rank = max(max_selected_rank, rank)
                used_tail = used_tail or 21 <= rank <= 25
                selected_from_tail += int(21 <= rank <= 25)
                evidence_verified = item.get("question_evidence_verified", True)
                has_unverified_evidence = (
                    has_unverified_evidence or evidence_verified is False
                )
                unverified_evidence_items += int(evidence_verified is False)
                selected_labels.append(
                    {
                        "label_id": label_id,
                        "label_name": label.get("label_name", ""),
                        "label_path": format_label_path(label.get("label_path")),
                        "candidate_rank": rank,
                        "sources": candidate.get("sources", []),
                        "sparse_rank": candidate.get("sparse_rank"),
                        "dense_rank": candidate.get("dense_rank"),
                        "evidence": item["question_evidence"],
                        "evidence_verified": evidence_verified,
                        "necessity": item["necessity"],
                    }
                )
            questions_using_tail += int(used_tail)
            questions_with_unverified_evidence += int(has_unverified_evidence)
            selected_count_distribution[str(len(selected_labels))] += 1
            need_expand += int(parsed["need_expand_recall"])
            none_count += int(parsed["none_of_candidates"])
            prediction = {
                "question_id": question_id,
                "parent_id": unit.get("parent_id", question_id),
                "unit_type": unit.get("unit_type", ""),
                "selected_labels": selected_labels,
                "none_of_candidates": parsed["none_of_candidates"],
                "need_expand_recall": parsed["need_expand_recall"],
                "reason": parsed["reason"],
                "needs_review": has_unverified_evidence,
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
    report = {
        "input": len(units),
        "processed": success,
        "success": success,
        "error": len(units) - success,
        "pending": len(units) - success,
        "evidence_rows": evidence_rows,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
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
        "unverified_evidence_items": unverified_evidence_items,
        "questions_with_unverified_evidence": questions_with_unverified_evidence,
        "model": model,
        "prompt_version": prompt_version,
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
