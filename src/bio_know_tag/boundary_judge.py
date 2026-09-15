"""Binary DS Judge for teacher-authored Label boundaries using legacy weak labels."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.ds import DSRequestError, append_evidence, parse_json_content
from bio_know_tag.legacy_validation import valid_legacy_targets


PROMPT_VERSION = "label-boundary-binary-v1"


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
    for unit in _read_jsonl(units_path):
        scanned += 1
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
    """Build a compact binary prompt without exposing historical IDs."""
    return f"""你是严格的高中生物知识点边界审核员。

判断完成当前设问时，是否直接需要使用下面这个Label。
只有题目的正确解答、对选项正误的必要判断，或解析的核心推理直接使用该Label时，才判true。
材料背景、实验工具、弱相关联想、上位概念或被distinctions排除的相近内容，判false。

Label名称：{label.get('label_name', '')}
definition：{label.get('definition', '')}
core_concepts：{label.get('core_concepts', '')}
common_assessments：{label.get('common_assessments', '')}
distinctions：{label.get('distinctions', '')}

父题公共材料（仅作当前小题语境）：{_clip(sample.get('parent_stem'), 1200)}
当前题干：{_clip(sample.get('stem'), 2400)}
选项：{_clip(sample.get('options'), 1600)}
答案：{_clip(sample.get('answer_text'), 1000)}
解析：{_clip(sample.get('analysis'), 3000)}

只输出JSON，不要解释：
{{"applicable":true}}
""".strip()


def validate_boundary_result(value: dict[str, Any]) -> dict[str, bool]:
    if set(value) != {"applicable"} or not isinstance(value.get("applicable"), bool):
        raise ValueError("response must contain only boolean applicable")
    return {"applicable": value["applicable"]}


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
    max_tokens: int = 32,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run resumable binary Judge requests for sampled question-Label pairs."""
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
            "stage": "label_boundary_binary_judge",
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
                    {"role": "system", "content": "你是严格的高中生物知识点边界审核员，只输出JSON。"},
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
    per_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in completed.values():
        label_id = str(record["label_id"])
        relation = str(record.get("expected_relation") or "")
        applicable = bool(record["parsed_response"]["applicable"])
        per_label_counts[label_id][f"{relation}_total"] += 1
        per_label_counts[label_id][f"{relation}_{str(applicable).lower()}"] += 1
    per_label_path = output_dir / "per_label.jsonl"
    with per_label_path.open("w", encoding="utf-8", newline="\n") as output:
        for label_id, counter in sorted(per_label_counts.items()):
            positive_total = counter["legacy_positive_total"]
            negative_total = counter["sibling_hard_negative_total"]
            row = {
                "label_id": label_id,
                "label_name": labels[label_id].get("label_name", ""),
                "legacy_positive_total": positive_total,
                "legacy_positive_true": counter["legacy_positive_true"],
                "legacy_positive_true_rate": round(counter["legacy_positive_true"] / positive_total, 6) if positive_total else None,
                "sibling_negative_total": negative_total,
                "sibling_negative_false": counter["sibling_hard_negative_false"],
                "sibling_negative_false_rate": round(counter["sibling_hard_negative_false"] / negative_total, 6) if negative_total else None,
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
                    "applicable": record["parsed_response"]["applicable"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
    return report


def _summarize(samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]], evidence_rows: int, model: str, workers: int) -> dict[str, Any]:
    positives = [record for record in completed.values() if record.get("expected_relation") == "legacy_positive"]
    negatives = [record for record in completed.values() if record.get("expected_relation") == "sibling_hard_negative"]
    positive_true = sum(record["parsed_response"]["applicable"] for record in positives)
    negative_false = sum(not record["parsed_response"]["applicable"] for record in negatives)
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
        "legacy_positive_completed": len(positives),
        "legacy_positive_true": positive_true,
        "legacy_positive_true_rate": round(positive_true / len(positives), 6) if positives else None,
        "sibling_negative_completed": len(negatives),
        "sibling_negative_false": negative_false,
        "sibling_negative_false_rate": round(negative_false / len(negatives), 6) if negatives else None,
        "ground_truth_warning": "Agreement with legacy weak labels is diagnostic, not gold accuracy.",
    }
