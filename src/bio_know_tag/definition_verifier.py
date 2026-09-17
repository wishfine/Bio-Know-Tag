"""Independent definition-only verification for selected candidate Labels."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.adjudication import (
    _ensure_run_manifest,
    _file_sha256,
    _read_jsonl,
    _write_json_atomic,
)
from bio_know_tag.ds import DSRequestError, append_evidence, parse_json_content
from bio_know_tag.retrieval import format_label_path


PROMPT_VERSION = "definition-verifier-v1-independent"
ORDER_VERSION = "definition-verifier-v1-order"


def build_definition_verifier_prompt(
    unit: dict[str, Any],
    stage1_prediction: dict[str, Any],
    labels_by_id: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, str]]:
    question_id = str(unit.get("question_id") or "")
    selected = list(stage1_prediction.get("selected_labels") or [])
    ordered = sorted(
        selected,
        key=lambda item: hashlib.sha256(
            f"{question_id}\0{item['label_id']}\0{ORDER_VERSION}".encode("utf-8")
        ).digest(),
    )
    code_map = {
        f"V{index:02d}": str(item["label_id"])
        for index, item in enumerate(ordered, 1)
    }
    label_cards = []
    for code, label_id in code_map.items():
        label = labels_by_id[label_id]
        label_cards.append(
            {
                "code": code,
                "label_name": label.get("label_name", ""),
                "label_path": format_label_path(label.get("label_path")),
                "definition": label.get("definition", ""),
                "distinctions": label.get("distinctions", ""),
            }
        )
    question = {
        "question_id": question_id,
        "unit_type": unit.get("unit_type", ""),
        "current_question": {
            "stem": str(unit.get("stem") or "")[:5000],
            "options": str(unit.get("options") or "")[:3000],
            "answer_text": str(unit.get("answer_text") or "")[:2000],
            "analysis": str(unit.get("analysis") or "")[:6000],
        },
        "parent_context_for_reference_only": str(unit.get("parent_stem") or "")[:3000],
    }
    prompt = f"""你是独立的高中生物Label定义复核器。你不负责召回Label，也不知道上一阶段为什么选它们。对每个候选独立判断，不得因第一阶段已选中而倾向于保留。

只看definition和distinctions界定Label范围；本任务故意不提供core_concepts、第一阶段reason、候选来源、排名和分数。

判定标准：
1. match=true仅当当前设问为得到答案而必须直接调用该Label定义范围。
2. 对象、任务、目标或层级任一不同，match=false。
3. 仅共享名词、底层机制、材料背景、父题主题、上下游步骤或一般方法，match=false。
4. Label定义若限定特定实验/方法/调查目标，题目必须实际考查该目标；仅考查后续产物、测定、应用或其他同类方法时为false。
5. 只有部分相符、边界不确定或需要用定义之外的常识扩张才能成立时，统一为false。
6. parent_context_for_reference_only只能解除“该实验/该物质/图中”等指代，不能替代当前设问制造考点。

题目：
{json.dumps(question, ensure_ascii=False)}

待独立复核的Label：
{json.dumps(label_cards, ensure_ascii=False)}

必须对每个code输出且只输出一次。question_target和definition_target各不超过60字，reason不超过80字。
只输出JSON：
{{
  "results": [
    {{
      "code": "V01",
      "match": false,
      "question_target": "当前设问真正考什么",
      "definition_target": "Label定义限定考什么",
      "reason": "为什么一致或不一致"
    }}
  ]
}}
不要输出Markdown或JSON之外的文字。"""
    return prompt, code_map


def validate_definition_verifier_result(
    value: dict[str, Any], known_codes: set[str]
) -> list[dict[str, Any]]:
    results = value.get("results")
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    normalized: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("result item must be an object")
        code = item.get("code")
        if not isinstance(code, str) or code not in known_codes:
            raise ValueError("result uses unknown code")
        if code in normalized:
            raise ValueError(f"duplicate result code: {code}")
        if not isinstance(item.get("match"), bool):
            raise ValueError(f"match must be boolean for {code}")
        record = {"code": code, "match": item["match"]}
        for field in ("question_target", "definition_target", "reason"):
            text = item.get(field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing {field} for {code}")
            text = text.strip()
            if len(text) > 500:
                raise ValueError(f"{field} is too long for {code}")
            record[field] = text
        normalized[code] = record
    missing = known_codes - set(normalized)
    if missing:
        raise ValueError(f"missing result code: {sorted(missing)[0]}")
    return [normalized[code] for code in sorted(normalized)]


def _latest_success(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    rows = 0
    if not path.exists():
        return latest, rows
    for record in _read_jsonl(path):
        rows += 1
        if (
            record.get("prompt_version") == PROMPT_VERSION
            and not record.get("error")
            and isinstance(record.get("parsed_response"), list)
        ):
            latest[str(record["question_id"])] = record
    return latest, rows


def run_definition_verifier(
    units_path: str | Path,
    stage1_predictions_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    limit: int | None = None,
    max_tokens: int = 1024,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.monotonic()
    units = _read_jsonl(units_path)
    if limit is not None:
        units = units[:limit]
    units_by_id = {str(row["question_id"]): row for row in units}
    predictions = _read_jsonl(stage1_predictions_path)
    predictions_by_id = {str(row["question_id"]): row for row in predictions}
    labels_by_id = {
        str(row["label_id"]): row for row in _read_jsonl(labels_path)
    }
    missing_predictions = set(units_by_id) - set(predictions_by_id)
    if missing_predictions:
        raise ValueError(
            f"missing stage1 prediction: {sorted(missing_predictions)[0]}"
        )
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.jsonl"
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "limit": limit,
        "max_tokens": max_tokens,
        "input_paths": {
            "units": str(Path(units_path)),
            "stage1_predictions": str(Path(stage1_predictions_path)),
            "labels": str(Path(labels_path)),
        },
        "input_sha256": {
            "units": _file_sha256(units_path),
            "stage1_predictions": _file_sha256(stage1_predictions_path),
            "labels": _file_sha256(labels_path),
        },
    }
    _ensure_run_manifest(output_dir / "run_manifest.json", manifest)
    completed, evidence_rows = _latest_success(evidence_path)
    pending = [
        row
        for row in units
        if (predictions_by_id[str(row["question_id"])].get("selected_labels") or [])
        and str(row["question_id"]) not in completed
    ]

    def verify(unit: dict[str, Any]) -> dict[str, Any]:
        question_id = str(unit["question_id"])
        prediction = predictions_by_id[question_id]
        prompt, code_map = build_definition_verifier_prompt(
            unit, prediction, labels_by_id
        )
        record = {
            "stage": "definition_verifier",
            "prompt_version": PROMPT_VERSION,
            "question_id": question_id,
            "candidate_code_map": code_map,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
            "usage": None,
            "retry_errors": [],
            "error": None,
        }
        try:
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": "你是独立的高中生物Label定义复核器，只输出JSON。",
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
                    "retry_errors": list(getattr(response, "retry_errors", ())),
                }
            )
            record["parsed_response"] = validate_definition_verifier_result(
                parse_json_content(response.content), set(code_map)
            )
        except DSRequestError as exc:
            record.update(
                {
                    "endpoint": exc.endpoint,
                    "attempts": exc.attempts,
                    "latency_seconds": exc.latency_seconds,
                    "retry_errors": list(exc.retry_errors),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    requests_succeeded = 0
    requests_failed = 0
    if workers == 1:
        results = map(verify, pending)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(verify, pending)
    try:
        for index, record in enumerate(results, 1):
            append_evidence(evidence_path, record)
            evidence_rows += 1
            requests_failed += int(bool(record["error"]))
            requests_succeeded += int(not record["error"])
            if not record["error"]:
                completed[str(record["question_id"])] = record
            print(
                f"[{index}/{len(pending)}] {record['question_id']} "
                f"{'ERROR' if record['error'] else 'OK'}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    completed, evidence_rows = _latest_success(evidence_path)
    output_path = output_dir / "predictions.jsonl"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    labels_kept = 0
    labels_rejected = 0
    questions_all_rejected = 0
    usable = 0
    rejected_distribution: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for unit in units:
            question_id = str(unit["question_id"])
            stage1 = predictions_by_id[question_id]
            selected = list(stage1.get("selected_labels") or [])
            if selected:
                record = completed.get(question_id)
                if not record:
                    continue
                code_map = record["candidate_code_map"]
                checks_by_label = {
                    code_map[item["code"]]: item
                    for item in record["parsed_response"]
                }
            else:
                checks_by_label = {}
            kept = []
            rejected = []
            for label in selected:
                label_id = str(label["label_id"])
                check = checks_by_label[label_id]
                enriched = {**label, "definition_verification": check}
                if check["match"]:
                    kept.append(enriched)
                    labels_kept += 1
                else:
                    rejected.append(enriched)
                    labels_rejected += 1
                    rejected_distribution[label.get("label_name", label_id)] += 1
            all_rejected = bool(selected and not kept)
            questions_all_rejected += int(all_rejected)
            usable_for_training = bool(
                stage1.get("usable_for_training") and kept and not all_rejected
            )
            usable += int(usable_for_training)
            verified = {
                **stage1,
                "stage1_reason": stage1.get("reason", ""),
                "reason": (
                    "独立定义复核后保留："
                    + "、".join(str(item.get("label_name") or "") for item in kept)
                    if kept
                    else "独立定义复核后未保留Label。"
                ),
                "selected_labels": kept,
                "definition_rejected_labels": rejected,
                "definition_verifier_all_rejected": all_rejected,
                "needs_review": bool(stage1.get("needs_review") or all_rejected),
                "usable_for_training": usable_for_training,
                "definition_verifier_version": PROMPT_VERSION,
            }
            output.write(json.dumps(verified, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    temporary.replace(output_path)
    report = {
        "input": len(units),
        "questions_with_selected_labels": sum(
            bool(predictions_by_id[qid].get("selected_labels"))
            for qid in units_by_id
        ),
        "processed": len(completed),
        "pending": len(pending) - requests_succeeded,
        "error": requests_failed,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "evidence_rows": evidence_rows,
        "labels_kept": labels_kept,
        "labels_rejected": labels_rejected,
        "questions_all_rejected": questions_all_rejected,
        "usable_for_training": usable,
        "rejected_label_distribution": dict(rejected_distribution.most_common()),
        "workers": workers,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "run_wall_seconds": round(time.monotonic() - started, 3),
        "input_sha256": manifest["input_sha256"],
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report
