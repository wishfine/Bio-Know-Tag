"""Scored DS Judge for teacher-authored Label coverage using legacy weak labels."""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.ds import DSRequestError, append_evidence, parse_json_content
from bio_know_tag.legacy_validation import valid_legacy_targets


PROMPT_VERSION = "label-definition-coverage-v3-boundary-calibrated"
MATCH_THRESHOLD = 0.70
DIFFERENCE_TYPES = {
    "matched",
    "legacy_label_wrong",
    "parent_union_not_child",
    "definition_too_narrow",
    "boundary_ambiguous",
    "related_but_not_direct",
    "context_insufficient",
    "invalid_question",
}


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _ensure_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("run manifest differs; use a new run directory")
        return
    _write_json_atomic(path, manifest)


def _load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    labels = {
        str(row["label_id"]): row
        for row in _read_jsonl(path)
        if str(row.get("label_id") or "")
    }
    if not labels:
        raise ValueError("current label catalog is empty")
    return labels


def _priority(seed: str, label_id: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{label_id}:{question_id}".encode()).hexdigest()


def _keep_best(
    buckets: dict[str, list[tuple[int, str, dict[str, Any]]]],
    label_id: str,
    unit: dict[str, Any],
    *,
    limit: int,
    seed: str,
) -> None:
    if limit == 0:
        return
    question_id = str(unit.get("question_id") or "")
    score = int(_priority(seed, label_id, question_id), 16)
    entry = (-score, question_id, unit)
    bucket = buckets[label_id]
    if len(bucket) < limit:
        heapq.heappush(bucket, entry)
    elif score < -bucket[0][0]:
        heapq.heapreplace(bucket, entry)


def build_boundary_sample(
    units_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    positive_per_label: int = 5,
    negative_per_label: int = 0,
    seed: str = "label-boundary-v1",
    unit_types: set[str] | None = None,
    image_context_path: str | Path | None = None,
    exclude_content_review: bool = False,
) -> dict[str, Any]:
    """Sample legacy positives and optional sibling-label hard negatives per Label."""
    if positive_per_label < 1 or negative_per_label < 0:
        raise ValueError("sample counts are invalid")
    labels = _load_labels(labels_path)
    label_ids = set(labels)
    parent_to_ids: dict[str, list[str]] = defaultdict(list)
    for label_id, label in labels.items():
        path = str(label.get("label_path") or "").replace("->", "@")
        parent_to_ids[path.rsplit("@", 1)[0]].append(label_id)
    siblings = {}
    for label_id, label in labels.items():
        path = str(label.get("label_path") or "").replace("->", "@")
        siblings[label_id] = set(parent_to_ids[path.rsplit("@", 1)[0]]) - {label_id}

    positives: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    negatives: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    scanned = obsolete_assignments = 0
    text_ineligible_skipped = content_review_skipped = 0
    if image_context_path is None:
        unit_rows = ((unit, None) for unit in _read_jsonl(units_path))
    else:
        unit_rows = itertools.zip_longest(
            _read_jsonl(units_path), _read_jsonl(image_context_path)
        )
    for unit, context in unit_rows:
        if unit is None or (image_context_path is not None and context is None):
            raise ValueError("units and image context must contain the same number of rows")
        scanned += 1
        if context is not None:
            question_id = str(unit.get("question_id") or "")
            if question_id != str(context.get("question_id") or ""):
                raise ValueError(f"units/image-context order mismatch at {question_id}")
            if not bool(context.get("eligible_for_text_labeling")):
                text_ineligible_skipped += 1
                continue
            if exclude_content_review and bool(context.get("needs_content_review")):
                content_review_skipped += 1
                continue
        if unit_types and str(unit.get("unit_type") or "") not in unit_types:
            continue
        valid, obsolete = valid_legacy_targets(unit, label_ids)
        obsolete_assignments += len(obsolete)
        question_id = str(unit.get("question_id") or "")
        valid_set = set(valid)
        for label_id in valid:
            _keep_best(
                positives,
                label_id,
                unit,
                limit=positive_per_label,
                seed=seed,
            )
        if negative_per_label:
            negative_targets = set().union(*(siblings[label_id] for label_id in valid)) if valid else set()
            for label_id in negative_targets - valid_set:
                _keep_best(
                    negatives,
                    label_id,
                    unit,
                    limit=negative_per_label,
                    seed=seed,
                )

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "boundary_samples.jsonl"
    rows = []
    for label_id in labels:
        for _, _, unit in sorted(positives[label_id], key=lambda item: (-item[0], item[1])):
            rows.append((label_id, "legacy_positive", unit))
        for _, _, unit in sorted(negatives[label_id], key=lambda item: (-item[0], item[1])):
            rows.append((label_id, "sibling_hard_negative", unit))
    rows.sort(key=lambda item: (item[0], item[1], str(item[2].get("question_id") or "")))
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for label_id, relation, unit in rows:
            question_id = str(unit.get("question_id") or "")
            row = {
                "pair_id": f"{question_id}::{label_id}",
                "question_id": question_id,
                "label_id": label_id,
                "expected_relation": relation,
                "unit_type": unit.get("unit_type"),
                "parent_stem": unit.get("parent_stem", ""),
                "stem": unit.get("stem", ""),
                "options": unit.get("options", ""),
                "answer_text": unit.get("answer_text", ""),
                "analysis": unit.get("analysis", ""),
                "flags": unit.get("flags") or {},
            }
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    relation_counts = Counter(relation for _, relation, _ in rows)
    labels_with_positive = sum(bool(positives[label_id]) for label_id in labels)
    report = {
        "units_scanned": scanned,
        "samples": len(rows),
        "positive_per_label": positive_per_label,
        "negative_per_label": negative_per_label,
        "relation_counts": dict(sorted(relation_counts.items())),
        "labels_with_legacy_positive": labels_with_positive,
        "labels_without_legacy_positive": len(labels) - labels_with_positive,
        "obsolete_legacy_assignments_ignored": obsolete_assignments,
        "image_context_path": str(image_context_path) if image_context_path else None,
        "text_ineligible_units_skipped": text_ineligible_skipped,
        "content_review_units_skipped": content_review_skipped,
        "exclude_content_review": exclude_content_review,
        "seed": seed,
        "unit_types": sorted(unit_types) if unit_types else ["all"],
        "ground_truth_warning": "Legacy positives and sibling negatives are weak supervision, not verified gold labels.",
    }
    _write_json_atomic(output_dir / "report.json", report)
    return report


def _clip(value: Any, size: int) -> str:
    text = str(value or "")
    if len(text) <= size:
        return text
    return f"{text[: size // 3]}…[中间省略]…{text[-(size - size // 3):]}"


def build_boundary_prompt(sample: dict[str, Any], label: dict[str, Any]) -> str:
    """Build a scored definition-coverage prompt without exposing historical IDs."""
    unit_type = str(sample.get("unit_type") or "unknown")
    return f"""你是严格的高中生物知识点释义覆盖审核员。

任务：判断“当前题目实际考查内容”与给定Label及老师释义的匹配程度。
只判断当前题目；父题材料只用于补全当前小题的指代和语境。

判定原则：
1. 先独立概括当前题目的实际考点，再核对Label；不要先看Label后从题中寻找牵强证据。
2. 只有正确解题、判断选项或解析核心推理直接需要该知识点，才算匹配。
3. 若题目直接考查definition或core_concepts明确包含的一个子主题，即可匹配；无需覆盖该Label释义中的全部组成部分，也不要求同时考查与其他概念的比较。
4. common_assessments中的常见考查方式是示例而不是穷举清单；题目未逐字命中示例，但实质属于该Label时仍可高分。
5. 必须同时核对知识对象和任务目标。仅共享实验动作、工具、方法或关键词，但实验对象、所属模块或实际考点不同，不算匹配。
6. 对名称或释义明确为“综合、整合、综合分析”的综合类Label，只有题目实际联动了释义或distinctions要求的多个方面时才匹配；只考一个原子知识点不能用综合Label兜底。
7. 材料背景、弱相关联想、仅时间/章节相邻、宽泛上位概念，不算匹配。
8. 对小题，父题级Label并不自动属于每个小题；若只属于父题或兄弟小题，应识别为parent_union_not_child。
9. 若Label名称和高中生物标准含义明显适合、但老师四字段遗漏了正常且重要的考查范围，应标为definition_too_narrow，而不是简单断言旧标签错误。
10. 不得因为历史上可能打过该标签而迁就；你看不到旧ID，只依据题目和老师给出的Label信息判断。

分数标准：
- 0.90-1.00：核心考点明确属于该Label，边界清楚。
- 0.80-0.89：明确匹配，仅有很小边界差异。
- 0.70-0.79：可以判为匹配，但存在边界或信息不完整风险。
- 0.40-0.69：有关联或释义疑似偏窄，但不足以确认匹配。
- 0.10-0.39：弱关联、背景关联或相近知识点。
- 0.00-0.09：基本无关；完全无关时给0。

Label名称：{label.get('label_name', '')}
definition：{label.get('definition', '')}
core_concepts：{label.get('core_concepts', '')}
common_assessments：{label.get('common_assessments', '')}
distinctions：{label.get('distinctions', '')}

题目单元类型：{unit_type}
父题公共材料（仅作当前小题语境）：{_clip(sample.get('parent_stem'), 1200)}
当前题干：{_clip(sample.get('stem'), 2400)}
选项：{_clip(sample.get('options'), 1600)}
答案：{_clip(sample.get('answer_text'), 1000)}
解析：{_clip(sample.get('analysis'), 3000)}

差异类型只能从以下值选择一个：
matched、legacy_label_wrong、parent_union_not_child、definition_too_narrow、boundary_ambiguous、related_but_not_direct、context_insufficient、invalid_question

只输出JSON，不要输出Markdown：
{{"score":0.92,"difference_type":"matched","reason":"一句话说明题目实际考点与释义是否覆盖"}}
""".strip()


def validate_boundary_result(value: dict[str, Any]) -> dict[str, Any]:
    required = {"score", "difference_type", "reason"}
    if set(value) != required:
        raise ValueError(f"response must contain exactly: {sorted(required)}")
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1")
    difference_type = value.get("difference_type")
    if difference_type not in DIFFERENCE_TYPES:
        raise ValueError(f"unknown difference_type: {difference_type}")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if len(reason) > 1000:
        raise ValueError("reason is too long")
    return {
        "score": round(score, 6),
        "match": score >= MATCH_THRESHOLD,
        "difference_type": difference_type,
        "reason": reason.strip(),
    }


def _latest_success(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    latest = {}
    rows = 0
    if not path.exists():
        return latest, rows
    for row in _read_jsonl(path):
        rows += 1
        if row.get("prompt_version") == PROMPT_VERSION and not row.get("error") and isinstance(row.get("parsed_response"), dict):
            latest[str(row["pair_id"])] = row
    return latest, rows


def run_boundary_judge(
    samples_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    max_tokens: int = 256,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run resumable scored Judge requests for sampled question-Label pairs."""
    if workers < 1:
        raise ValueError("workers must be positive")
    samples = list(_read_jsonl(samples_path))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    labels = _load_labels(labels_path)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "max_tokens": max_tokens,
        "limit": limit,
        "input_paths": {
            "samples": str(Path(samples_path)),
            "labels": str(Path(labels_path)),
        },
        "input_sha256": {
            "samples": _file_sha256(samples_path),
            "labels": _file_sha256(labels_path),
        },
    }
    _ensure_manifest(output_dir / "run_manifest.json", manifest)
    evidence_path = output_dir / "evidence.jsonl"
    completed, evidence_rows = _latest_success(evidence_path)
    pending = [sample for sample in samples if str(sample["pair_id"]) not in completed]

    def judge(sample: dict[str, Any]) -> dict[str, Any]:
        pair_id = str(sample["pair_id"])
        label_id = str(sample["label_id"])
        if label_id not in labels:
            raise ValueError(f"sample uses obsolete or unknown label_id: {label_id}")
        prompt = build_boundary_prompt(sample, labels[label_id])
        record = {
            "stage": "label_definition_coverage_judge",
            "prompt_version": PROMPT_VERSION,
            "pair_id": pair_id,
            "question_id": str(sample["question_id"]),
            "label_id": label_id,
            "expected_relation": sample.get("expected_relation"),
            "unit_type": sample.get("unit_type"),
            "model": model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_chars": len(prompt),
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
                    {"role": "system", "content": "你是严格的高中生物知识点释义覆盖审核员，只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            record.update(
                raw_response=response.content,
                endpoint=response.endpoint,
                attempts=response.attempts,
                latency_seconds=response.latency_seconds,
                usage=getattr(response, "usage", None),
                retry_errors=list(getattr(response, "retry_errors", ())),
            )
            record["parsed_response"] = validate_boundary_result(parse_json_content(response.content))
        except DSRequestError as exc:
            record.update(endpoint=exc.endpoint, attempts=exc.attempts, latency_seconds=exc.latency_seconds, retry_errors=list(exc.retry_errors))
            record["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    iterator = map(judge, pending) if workers == 1 else None
    executor = None
    if iterator is None:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(judge, pending)
    try:
        for index, record in enumerate(iterator, 1):
            append_evidence(evidence_path, record)
            evidence_rows += 1
            if not record["error"]:
                completed[str(record["pair_id"])] = record
            print(f"[{index}/{len(pending)}] {record['pair_id']} {'ERROR' if record['error'] else 'OK'}", flush=True)
            _write_json_atomic(output_dir / "report.json", _summarize(samples, completed, evidence_rows, model, workers))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    completed, evidence_rows = _latest_success(evidence_path)
    report = _summarize(samples, completed, evidence_rows, model, workers)
    _write_json_atomic(output_dir / "report.json", report)
    per_label_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed.values():
        per_label_records[str(record["label_id"])].append(record)
    per_label_path = output_dir / "per_label.jsonl"
    with per_label_path.open("w", encoding="utf-8", newline="\n") as output:
        for label_id, records in sorted(per_label_records.items()):
            positives = [r for r in records if r.get("expected_relation") == "legacy_positive"]
            negatives = [r for r in records if r.get("expected_relation") == "sibling_hard_negative"]
            positive_scores = [float(r["parsed_response"]["score"]) for r in positives]
            negative_scores = [float(r["parsed_response"]["score"]) for r in negatives]
            positive_match = sum(bool(r["parsed_response"]["match"]) for r in positives)
            negative_nonmatch = sum(not bool(r["parsed_response"]["match"]) for r in negatives)
            row = {
                "label_id": label_id,
                "label_name": labels[label_id].get("label_name", ""),
                "legacy_positive_total": len(positives),
                "legacy_positive_match": positive_match,
                "legacy_positive_match_rate": round(positive_match / len(positives), 6) if positives else None,
                "legacy_positive_mean_score": round(sum(positive_scores) / len(positive_scores), 6) if positive_scores else None,
                "legacy_positive_zero_score": sum(score == 0 for score in positive_scores),
                "legacy_positive_gray_zone": sum(0.40 <= score < MATCH_THRESHOLD for score in positive_scores),
                "legacy_positive_score_distribution": dict(sorted(Counter(_score_bucket(score) for score in positive_scores).items())),
                "legacy_positive_difference_types": dict(sorted(Counter(r["parsed_response"]["difference_type"] for r in positives).items())),
                "sibling_negative_total": len(negatives),
                "sibling_negative_nonmatch": negative_nonmatch,
                "sibling_negative_nonmatch_rate": round(negative_nonmatch / len(negatives), 6) if negatives else None,
                "sibling_negative_mean_score": round(sum(negative_scores) / len(negative_scores), 6) if negative_scores else None,
            }
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for sample in samples:
            record = completed.get(str(sample["pair_id"]))
            if record:
                output.write(json.dumps({
                    "pair_id": record["pair_id"],
                    "question_id": record["question_id"],
                    "label_id": record["label_id"],
                    "expected_relation": record["expected_relation"],
                    "unit_type": record.get("unit_type"),
                    "score": record["parsed_response"]["score"],
                    "match": record["parsed_response"]["match"],
                    "difference_type": record["parsed_response"]["difference_type"],
                    "reason": record["parsed_response"]["reason"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def _score_bucket(score: float) -> str:
    if score >= 0.90:
        return ">=0.90"
    if score >= 0.80:
        return "0.80-0.89"
    if score >= 0.70:
        return "0.70-0.79"
    if score >= 0.40:
        return "0.40-0.69"
    if score >= 0.30:
        return "0.30-0.39"
    if score >= 0.20:
        return "0.20-0.29"
    if score >= 0.10:
        return "0.10-0.19"
    if score > 0:
        return "0.01-0.09"
    return "0.00"


def _summarize(samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]], evidence_rows: int, model: str, workers: int) -> dict[str, Any]:
    positives = [record for record in completed.values() if record.get("expected_relation") == "legacy_positive"]
    negatives = [record for record in completed.values() if record.get("expected_relation") == "sibling_hard_negative"]
    positive_match = sum(record["parsed_response"]["match"] for record in positives)
    negative_nonmatch = sum(not record["parsed_response"]["match"] for record in negatives)
    all_records = list(completed.values())
    score_distribution = Counter(_score_bucket(float(record["parsed_response"]["score"])) for record in all_records)
    difference_types = Counter(record["parsed_response"]["difference_type"] for record in all_records)
    by_unit_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in positives:
        grouped[str(record.get("unit_type") or "unknown")].append(record)
    for unit_type, records in sorted(grouped.items()):
        matches = sum(record["parsed_response"]["match"] for record in records)
        by_unit_type[unit_type] = {
            "legacy_positive_total": len(records),
            "match": matches,
            "match_rate": round(matches / len(records), 6),
            "mean_score": round(sum(record["parsed_response"]["score"] for record in records) / len(records), 6),
        }
    question_scores: dict[str, list[float]] = defaultdict(list)
    for record in positives:
        question_scores[str(record["question_id"])].append(float(record["parsed_response"]["score"]))
    question_grades: Counter[str] = Counter()
    for scores in question_scores.values():
        best = max(scores)
        if best >= 0.80:
            question_grades["A_>=0.80"] += 1
        elif best >= MATCH_THRESHOLD:
            question_grades["B_0.70-0.79"] += 1
        elif best >= 0.40:
            question_grades["C_0.40-0.69"] += 1
        elif best == 0:
            question_grades["D_zero"] += 1
        else:
            question_grades["D_0.01-0.39"] += 1
    success = len(completed)
    return {
        "input": len(samples),
        "processed": success,
        "success": success,
        "error": len(samples) - success,
        "pending": len(samples) - success,
        "evidence_rows": evidence_rows,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "workers": workers,
        "match_threshold": MATCH_THRESHOLD,
        "score_distribution": dict(sorted(score_distribution.items())),
        "gray_zone_0.40_0.69": sum(0.40 <= record["parsed_response"]["score"] < MATCH_THRESHOLD for record in all_records),
        "difference_type_counts": dict(sorted(difference_types.items())),
        "legacy_positive_completed": len(positives),
        "legacy_positive_match": positive_match,
        "legacy_positive_match_rate": round(positive_match / len(positives), 6) if positives else None,
        "legacy_positive_score_distribution": dict(sorted(Counter(_score_bucket(float(record["parsed_response"]["score"])) for record in positives).items())),
        "legacy_positive_by_unit_type": by_unit_type,
        "unique_questions_with_legacy_positive": len(question_scores),
        "question_training_grade_distribution": dict(sorted(question_grades.items())),
        "sibling_negative_completed": len(negatives),
        "sibling_negative_nonmatch": negative_nonmatch,
        "sibling_negative_nonmatch_rate": round(negative_nonmatch / len(negatives), 6) if negatives else None,
        "ground_truth_warning": "Agreement with legacy weak labels is diagnostic, not gold accuracy.",
    }
