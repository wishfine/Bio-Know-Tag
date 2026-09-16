"""Mentor-compatible batched DS judging for Label-definition coverage."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bio_know_tag.ds import DSRequestError, parse_json_content


PROMPT_VERSION = "label-definition-coverage-mentor-batch-v1"
MATCH_THRESHOLD = 0.70


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                yield value


def _question_context(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": task.get("unit_type"),
        "parent_stem": task.get("parent_stem", ""),
        "stem": task.get("stem", ""),
        "options": task.get("options", ""),
        "answer": task.get("answer_text", ""),
        "analysis": task.get("analysis", ""),
    }


def build_batch_prompt(tasks: list[dict[str, Any]], label: dict[str, Any]) -> str:
    """Build the biology adaptation of the mentor's history scoring prompt."""
    if not tasks:
        raise ValueError("batch must not be empty")
    label_ids = {str(task.get("label_id") or "") for task in tasks}
    if label_ids != {str(label.get("label_id") or "")}:
        raise ValueError("every batch must contain exactly one matching label")
    teacher = {
        "definition": label.get("definition", ""),
        "core_concepts": label.get("core_concepts", ""),
        "common_assessments": label.get("common_assessments", ""),
        "distinctions": label.get("distinctions", ""),
    }
    questions = [
        {
            "task_id": str(task["pair_id"]),
            "question_id": str(task["question_id"]),
            "question_context": _question_context(task),
        }
        for task in tasks
    ]
    return f"""你是高中生物知识点标注审核专家。请根据教研给出的知识点释义，逐条判断目标题目是否考查该知识点，并输出匹配度。

判定规则：
1. 以目标小题或独立题的实际考查内容为准，不以背景材料中仅仅出现的词语为准。
2. 对 question_type=sub_question 或 orphan_sub_question 的任务，必须只判断当前小题；父题公共材料只用于理解语境，不能把父题或其他小题考查的知识点算到当前小题上。
3. 若目标题目仅把知识点作为背景、实验工具、材料来源或弱相关概念，而非主要考查对象，应给低分。
4. common_assessments是常见例子而非穷举；实际考点落入definition或core_concepts即可匹配。
5. match必须与阈值一致：relevance_score >= 0.70时为true。

匹配度标准：
- 0.90-1.00：目标题目的核心设问、材料或选项直接考查该知识点的核心内容。
- 0.70-0.89：目标题目主要考查该知识点，但只涉及部分内容、较间接或需要结合材料推断。
- 0.40-0.69：目标题目与知识点有明显关联，但知识点不是主要考查对象，或仅作为背景。
- 0.10-0.39：目标题目仅有弱关联、个别词面重合或背景重合。
- 0.00-0.09：目标题目与该知识点基本无关。

输出格式：
只输出严格JSON，不要Markdown、解释或额外文本。格式为：
{{"results":[{{"task_id":"","question_id":"","match":true,"relevance_score":0.92}}]}}

硬性要求：
- results 数量必须等于输入题目数量 {len(tasks)}。
- 顺序必须与输入题目顺序一致。
- 不得省略、合并或新增题目。
- relevance_score必须是0到1之间的数字，保留两位小数。
- 不要输出证据、理由或其他字段。

Label名称：{label.get('label_name', '')}

教研对该知识点的解释与说明：
{json.dumps(teacher, ensure_ascii=False, indent=2)}

待判定题目：
{json.dumps(questions, ensure_ascii=False, indent=2)}
""".strip()


def build_batches(
    tasks: list[dict[str, Any]], *, max_batch_size: int, char_budget: int
) -> list[list[dict[str, Any]]]:
    if max_batch_size < 1 or char_budget < 1:
        raise ValueError("batch limits must be positive")
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_order: list[str] = []
    for task in tasks:
        label_id = str(task.get("label_id") or "")
        if label_id not in by_label:
            label_order.append(label_id)
        by_label[label_id].append(task)
    batches: list[list[dict[str, Any]]] = []
    for label_id in label_order:
        current: list[dict[str, Any]] = []
        current_chars = 0
        for task in by_label[label_id]:
            task_chars = len(
                json.dumps(_question_context(task), ensure_ascii=False, separators=(",", ":"))
            )
            if current and (
                len(current) >= max_batch_size
                or current_chars + task_chars > char_budget
            ):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(task)
            current_chars += task_chars
        if current:
            batches.append(current)
    return batches


def normalize_batch_response(
    value: dict[str, Any], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(tasks):
        raise ValueError("results count mismatch")
    normalized: list[dict[str, Any]] = []
    for result, task in zip(results, tasks):
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
        task_id = str(result.get("task_id") or "")
        question_id = str(result.get("question_id") or "")
        if task_id != str(task["pair_id"]) or question_id != str(task["question_id"]):
            raise ValueError(f"task_id/question_id mismatch: {task_id}, {question_id}")
        score = result.get("relevance_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("relevance_score must be numeric")
        score = round(float(score), 2)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid relevance_score: {score}")
        normalized.append(
            {
                "task_id": task_id,
                "label_id": str(task["label_id"]),
                "question_type": task.get("unit_type"),
                "question_id": question_id,
                "match": score >= MATCH_THRESHOLD,
                "relevance_score": score,
            }
        )
    return normalized


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    if path.exists():
        for row in _read_jsonl(path):
            latest[str(row["task_id"])] = row
    return latest


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


def _summarize(
    tasks: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    *,
    batch_count: int,
    failed_batches: int,
    workers: int,
) -> dict[str, Any]:
    rows = list(completed.values())
    matches = sum(bool(row["match"]) for row in rows)
    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question_type") or "unknown")].append(row)
    for question_type, values in sorted(grouped.items()):
        hit = sum(bool(row["match"]) for row in values)
        by_type[question_type] = {
            "total": len(values),
            "match": hit,
            "match_rate": round(hit / len(values), 6),
        }
    return {
        "input": len(tasks),
        "processed": len(rows),
        "success": len(rows),
        "pending": len(tasks) - len(rows),
        "error": len(tasks) - len(rows),
        "prompt_version": PROMPT_VERSION,
        "match_threshold": MATCH_THRESHOLD,
        "match": matches,
        "match_rate": round(matches / len(rows), 6) if rows else None,
        "score_distribution": dict(
            sorted(Counter(_score_bucket(float(row["relevance_score"])) for row in rows).items())
        ),
        "by_question_type": by_type,
        "batches": batch_count,
        "failed_batches": failed_batches,
        "workers": workers,
        "ground_truth_warning": "Historical IDs are weak supervision, not verified gold labels.",
    }


def run_coverage_batches(
    tasks_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    workers: int = 8,
    max_batch_size: int = 40,
    char_budget: int = 55_000,
    max_tokens: int = 7_000,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run resumable same-Label batches and write compact task-level scores."""
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = list(_read_jsonl(tasks_path))
    if limit is not None:
        tasks = tasks[:limit]
    labels = {
        str(row["label_id"]): row
        for row in _read_jsonl(labels_path)
        if str(row.get("label_id") or "")
    }
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "workers": workers,
        "max_batch_size": max_batch_size,
        "char_budget": char_budget,
        "max_tokens": max_tokens,
        "limit": limit,
        "input_sha256": {
            "tasks": _file_sha256(tasks_path),
            "labels": _file_sha256(labels_path),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("run manifest differs; use a new run directory")
    else:
        _write_json_atomic(manifest_path, manifest)

    results_path = output_dir / "results.jsonl"
    evidence_path = output_dir / "evidence.jsonl"
    completed = _load_latest_results(results_path)
    pending = [task for task in tasks if str(task["pair_id"]) not in completed]
    batches = build_batches(
        pending, max_batch_size=max_batch_size, char_budget=char_budget
    )
    write_lock = threading.Lock()
    failed_batches = 0

    def judge_once(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        label_id = str(batch[0]["label_id"])
        if label_id not in labels:
            raise ValueError(f"unknown label_id: {label_id}")
        prompt = build_batch_prompt(batch, labels[label_id])
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            response = client.chat(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens
            )
            normalized = normalize_batch_response(
                parse_json_content(response.content), batch
            )
            evidence = {
                "prompt_version": PROMPT_VERSION,
                "label_id": label_id,
                "task_ids": [str(task["pair_id"]) for task in batch],
                "batch_size": len(batch),
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_response": response.content,
                "endpoint": response.endpoint,
                "attempts": response.attempts,
                "latency_seconds": response.latency_seconds,
                "usage": response.usage,
                "retry_errors": list(response.retry_errors),
                "created_at": created_at,
                "error": None,
            }
            return normalized, evidence
        except DSRequestError as exc:
            evidence = {
                "prompt_version": PROMPT_VERSION,
                "label_id": label_id,
                "task_ids": [str(task["pair_id"]) for task in batch],
                "batch_size": len(batch),
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_response": None,
                "endpoint": exc.endpoint,
                "attempts": exc.attempts,
                "latency_seconds": exc.latency_seconds,
                "usage": None,
                "retry_errors": list(exc.retry_errors),
                "created_at": created_at,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return [], evidence
        except Exception as exc:
            return [], {
                "prompt_version": PROMPT_VERSION,
                "label_id": label_id,
                "task_ids": [str(task["pair_id"]) for task in batch],
                "batch_size": len(batch),
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_response": locals().get("response").content if "response" in locals() else None,
                "endpoint": getattr(locals().get("response"), "endpoint", None),
                "attempts": getattr(locals().get("response"), "attempts", 0),
                "latency_seconds": getattr(locals().get("response"), "latency_seconds", None),
                "usage": getattr(locals().get("response"), "usage", None),
                "retry_errors": list(getattr(locals().get("response"), "retry_errors", ())),
                "created_at": created_at,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def judge(
        batch: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Judge one batch, bisecting deterministic request/format failures."""
        normalized, evidence = judge_once(batch)
        if normalized:
            evidence["adaptive_split"] = False
            return normalized, [evidence], 0

        error = str(evidence.get("error") or "")
        should_split = len(batch) > 1 and (
            error.startswith("ValueError:") or "HTTP Error 400" in error
        )
        evidence["adaptive_split"] = should_split
        if not should_split:
            return [], [evidence], 1

        midpoint = len(batch) // 2
        left_results, left_evidence, left_failures = judge(batch[:midpoint])
        right_results, right_evidence, right_failures = judge(batch[midpoint:])
        return (
            left_results + right_results,
            [evidence, *left_evidence, *right_evidence],
            left_failures + right_failures,
        )

    finished_batches = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge, batch): batch for batch in batches}
        for future in as_completed(futures):
            normalized, evidence_rows, leaf_failures = future.result()
            with write_lock:
                with evidence_path.open("a", encoding="utf-8") as output:
                    for evidence in evidence_rows:
                        output.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                if normalized:
                    with results_path.open("a", encoding="utf-8") as output:
                        for result in normalized:
                            result["model"] = model
                            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                            completed[str(result["task_id"])] = result
                failed_batches += leaf_failures
            finished_batches += 1
            print(
                f"batch {finished_batches}/{len(batches)} tasks={len(completed)}/{len(tasks)} "
                f"failed_batches={failed_batches}",
                flush=True,
            )

    completed = _load_latest_results(results_path)
    report = _summarize(
        tasks,
        completed,
        batch_count=len(batches),
        failed_batches=failed_batches,
        workers=workers,
    )
    _write_json_atomic(output_dir / "report.json", report)
    per_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed.values():
        per_label[str(row["label_id"])].append(row)
    with (output_dir / "per_label.jsonl").open("w", encoding="utf-8") as output:
        for label_id, rows in sorted(per_label.items()):
            matches = sum(bool(row["match"]) for row in rows)
            scores = [float(row["relevance_score"]) for row in rows]
            output.write(
                json.dumps(
                    {
                        "label_id": label_id,
                        "label_name": labels[label_id].get("label_name", ""),
                        "total": len(rows),
                        "match": matches,
                        "match_rate": round(matches / len(rows), 6),
                        "mean_score": round(sum(scores) / len(scores), 6),
                        "zero_score": sum(score == 0 for score in scores),
                        "gray_zone_0.40_0.69": sum(0.40 <= score < MATCH_THRESHOLD for score in scores),
                        "score_distribution": dict(
                            sorted(Counter(_score_bucket(score) for score in scores).items())
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return report
