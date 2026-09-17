"""Second-stage adjudication of accepted sibling-Label hard negatives."""

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


PROMPT_VERSION = "co-label-adjudication-v1-minimal-sufficient"
DECISIONS = (
    "合理共标",
    "目标Label边界过宽",
    "来源Label不足以描述该题",
    "无法判断",
)


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                yield value


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


def build_judge_tasks(
    samples: dict[str, dict[str, Any]],
    first_stage_results: dict[str, dict[str, Any]],
    *,
    threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """Keep only first-stage target acceptances for co-label adjudication."""
    tasks = []
    for task_id, sample in samples.items():
        result = first_stage_results.get(task_id)
        if result is None or float(result.get("relevance_score", -1)) < threshold:
            continue
        tasks.append(
            {
                **sample,
                "first_stage_score": float(result["relevance_score"]),
            }
        )
    tasks.sort(key=lambda row: (str(row["label_id"]), str(row["question_id"])))
    return tasks


def _label_card(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_id": str(label.get("label_id") or ""),
        "label_name": label.get("label_name", ""),
        "definition": label.get("definition", ""),
        "core_concepts": label.get("core_concepts", ""),
        "common_assessments": label.get("common_assessments", ""),
        "distinctions": label.get("distinctions", ""),
    }


def _question_context(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": task.get("unit_type"),
        "parent_stem": task.get("parent_stem", ""),
        "stem": task.get("stem", ""),
        "options": task.get("options", ""),
        "answer": task.get("answer_text", ""),
        "analysis": task.get("analysis", ""),
    }


def build_batch_prompt(
    tasks: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> str:
    if not tasks:
        raise ValueError("batch must not be empty")
    target_ids = {str(task["label_id"]) for task in tasks}
    if len(target_ids) != 1:
        raise ValueError("one batch must contain exactly one target Label")
    target_id = next(iter(target_ids))
    if target_id not in labels:
        raise ValueError(f"unknown target label: {target_id}")
    questions = []
    for task in tasks:
        source_ids = [str(value) for value in task.get("source_label_ids") or []]
        if not source_ids or any(source_id not in labels for source_id in source_ids):
            raise ValueError("unknown or empty source labels")
        questions.append(
            {
                "task_id": str(task["pair_id"]),
                "question_id": str(task["question_id"]),
                "source_labels": [_label_card(labels[source_id]) for source_id in source_ids],
                "question": _question_context(task),
            }
        )
    decisions = " | ".join(DECISIONS)
    return f"""你是高中生物知识点最终标注审核专家。

任务：题目已有一个或多个高置信来源Label；第一阶段又认为目标Label与题目匹配。现在请根据“最小充分知识点集合”，判断来源Label和目标Label是否应同时成为该题的最终标签。

判定原则：
1. 只看正确完成当前设问必须调用的知识；材料背景、工具、错误选项和弱相关内容不打标。
2. 上位Label不得仅因包含当前知识就共标；上位+下位机械重复时，优先最小、直接的Label。
3. 综合Label必须真正要求多个子模块联动；只考一个子模块不共标综合Label。
4. 比较/区别与联系Label必须真正要求比较或联系双方；只考一端不命中。
5. 实验Label只在实验目的、步骤、变量、现象、误差或方案评价是作答对象时命中。
6. 不要因为目标Label更宽泛、“也能解释”或与来源Label相邻就判为合理共标。
7. 不参考历史未打目标Label这一事实；仅根据题目和Label释义判断。

四类结论：
- 合理共标：来源Label与目标Label分别描述题目中不可替代的主要考点，两者都应保留。
- 目标Label边界过宽：来源Label已足以描述作答所需知识；目标Label只是上位包含、背景、弱相关，或未满足综合/比较条件。
- 来源Label不足以描述该题：目标Label才是主要或必要考点，来源Label组合本身不完整或口径不准。
- 无法判断：图片/关键上下文缺失，或现有释义不足以唯一判定。

硬性输出：
- 只输出严格JSON，不要Markdown、理由或额外文本。
- results数量必须等于{len(tasks)}，且顺序与输入一致。
- decision只能是：{decisions}

输出格式：
{{"results":[{{"task_id":"","question_id":"","decision":"合理共标"}}]}}

目标Label：
{json.dumps(_label_card(labels[target_id]), ensure_ascii=False, indent=2)}

待审核题目：
{json.dumps(questions, ensure_ascii=False, indent=2)}
""".strip()


def normalize_batch_response(
    value: dict[str, Any], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(tasks):
        raise ValueError("results count mismatch")
    normalized = []
    for result, task in zip(results, tasks):
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
        task_id = str(result.get("task_id") or "")
        question_id = str(result.get("question_id") or "")
        if task_id != str(task["pair_id"]) or question_id != str(task["question_id"]):
            raise ValueError("task_id/question_id mismatch")
        decision = str(result.get("decision") or "")
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision: {decision}")
        normalized.append(
            {
                "task_id": task_id,
                "question_id": question_id,
                "target_label_id": str(task["label_id"]),
                "source_label_ids": [str(value) for value in task["source_label_ids"]],
                "decision": decision,
            }
        )
    return normalized


def build_batches(
    tasks: list[dict[str, Any]],
    *,
    max_batch_size: int,
    char_budget: int,
    labels: dict[str, dict[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    if max_batch_size < 1 or char_budget < 1:
        raise ValueError("batch limits must be positive")
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order = []
    for task in tasks:
        target = str(task["label_id"])
        if target not in by_target:
            order.append(target)
        by_target[target].append(task)
    batches = []
    for target in order:
        current = []
        current_chars = 0
        for task in by_target[target]:
            task_chars = len(json.dumps(task, ensure_ascii=False, separators=(",", ":")))
            candidate = [*current, task]
            exceeds_budget = (
                len(build_batch_prompt(candidate, labels)) > char_budget
                if labels is not None
                else current_chars + task_chars > char_budget
            )
            if current and (len(current) >= max_batch_size or exceeds_budget):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(task)
            current_chars += task_chars
        if current:
            batches.append(current)
    return batches


def _load_latest(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in _read_jsonl(path)} if path.exists() else {}


def _summarize(
    tasks: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    *,
    batches: int,
    failed_batches: int,
    workers: int,
) -> dict[str, Any]:
    counts = Counter(str(row["decision"]) for row in completed.values())
    resolved = len(completed) - counts["无法判断"]
    true_errors = counts["目标Label边界过宽"]
    return {
        "input": len(tasks),
        "processed": len(completed),
        "success": len(completed),
        "pending": len(tasks) - len(completed),
        "error": len(tasks) - len(completed),
        "prompt_version": PROMPT_VERSION,
        "decision_counts": dict(sorted(counts.items())),
        "resolved": resolved,
        "true_boundary_errors": true_errors,
        "true_boundary_error_rate": round(true_errors / resolved, 6) if resolved else None,
        "batches": batches,
        "failed_batches": failed_batches,
        "workers": workers,
    }


def _write_summaries(
    output_dir: Path,
    tasks_by_id: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> None:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task_id, row in completed.items():
        task = tasks_by_id[task_id]
        target = str(task["label_id"])
        by_target[target].append(row)
        for source in task["source_label_ids"]:
            by_pair[(str(source), target)].append(row)
    with (output_dir / "per_target_label.jsonl").open("w", encoding="utf-8") as output:
        for target, rows in sorted(by_target.items()):
            counts = Counter(str(row["decision"]) for row in rows)
            resolved = len(rows) - counts["无法判断"]
            errors = counts["目标Label边界过宽"]
            output.write(
                json.dumps(
                    {
                        "target_label_id": target,
                        "target_label_name": labels[target].get("label_name", ""),
                        "total": len(rows),
                        "decision_counts": dict(sorted(counts.items())),
                        "resolved": resolved,
                        "true_boundary_errors": errors,
                        "true_boundary_error_rate": round(errors / resolved, 6)
                        if resolved
                        else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    with (output_dir / "source_target_decisions.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for (source, target), rows in sorted(by_pair.items()):
            counts = Counter(str(row["decision"]) for row in rows)
            output.write(
                json.dumps(
                    {
                        "source_label_id": source,
                        "source_label_name": labels[source].get("label_name", ""),
                        "target_label_id": target,
                        "target_label_name": labels[target].get("label_name", ""),
                        "total": len(rows),
                        "decision_counts": dict(sorted(counts.items())),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def run_co_label_judge(
    samples_path: str | Path,
    first_stage_results_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    workers: int = 8,
    max_batch_size: int = 20,
    char_budget: int = 45_000,
    max_tokens: int = 3_000,
    limit: int | None = None,
) -> dict[str, Any]:
    samples = {str(row["pair_id"]): row for row in _read_jsonl(samples_path)}
    first_stage = {
        str(row["task_id"]): row for row in _read_jsonl(first_stage_results_path)
    }
    labels = {str(row["label_id"]): row for row in _read_jsonl(labels_path)}
    tasks = build_judge_tasks(samples, first_stage)
    if limit is not None:
        tasks = tasks[:limit]
    tasks_by_id = {str(task["pair_id"]): task for task in tasks}
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
            "samples": _file_sha256(samples_path),
            "first_stage_results": _file_sha256(first_stage_results_path),
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
    completed = _load_latest(results_path)
    pending = [task for task in tasks if str(task["pair_id"]) not in completed]
    batches = build_batches(
        pending,
        max_batch_size=max_batch_size,
        char_budget=char_budget,
        labels=labels,
    )
    lock = threading.Lock()
    failed_batches = 0

    def judge_once(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = build_batch_prompt(batch, labels)
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            response = client.chat(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens
            )
            normalized = normalize_batch_response(
                parse_json_content(response.content), batch
            )
            return normalized, {
                "prompt_version": PROMPT_VERSION,
                "target_label_id": str(batch[0]["label_id"]),
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
        except DSRequestError as exc:
            return [], {
                "prompt_version": PROMPT_VERSION,
                "target_label_id": str(batch[0]["label_id"]),
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
        except Exception as exc:
            response = locals().get("response")
            return [], {
                "prompt_version": PROMPT_VERSION,
                "target_label_id": str(batch[0]["label_id"]),
                "task_ids": [str(task["pair_id"]) for task in batch],
                "batch_size": len(batch),
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_response": getattr(response, "content", None),
                "endpoint": getattr(response, "endpoint", None),
                "attempts": getattr(response, "attempts", 0),
                "latency_seconds": getattr(response, "latency_seconds", None),
                "usage": getattr(response, "usage", None),
                "retry_errors": list(getattr(response, "retry_errors", ())),
                "created_at": created_at,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def judge(
        batch: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        normalized, evidence = judge_once(batch)
        if normalized:
            evidence["adaptive_split"] = False
            return normalized, [evidence], 0
        error = str(evidence.get("error") or "")
        split = len(batch) > 1 and (
            error.startswith("ValueError:") or "HTTP Error 400" in error
        )
        evidence["adaptive_split"] = split
        if not split:
            return [], [evidence], 1
        midpoint = len(batch) // 2
        left_rows, left_evidence, left_failures = judge(batch[:midpoint])
        right_rows, right_evidence, right_failures = judge(batch[midpoint:])
        return (
            left_rows + right_rows,
            [evidence, *left_evidence, *right_evidence],
            left_failures + right_failures,
        )

    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge, batch): batch for batch in batches}
        for future in as_completed(futures):
            normalized, evidence_rows, leaf_failures = future.result()
            with lock:
                with evidence_path.open("a", encoding="utf-8") as output:
                    for evidence in evidence_rows:
                        output.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                if normalized:
                    with results_path.open("a", encoding="utf-8") as output:
                        for row in normalized:
                            row["model"] = model
                            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                            completed[row["task_id"]] = row
                failed_batches += leaf_failures
            finished += 1
            print(
                f"batch {finished}/{len(batches)} tasks={len(completed)}/{len(tasks)} "
                f"failed_batches={failed_batches}",
                flush=True,
            )

    completed = _load_latest(results_path)
    report = _summarize(
        tasks,
        completed,
        batches=len(batches),
        failed_batches=failed_batches,
        workers=workers,
    )
    _write_json_atomic(output_dir / "report.json", report)
    _write_summaries(output_dir, tasks_by_id, completed, labels)
    return report


def _corrected_screen(
    positive_count: int,
    positive_rate: float | None,
    valid_negative_count: int,
    corrected_error_rate: float | None,
) -> str:
    if positive_count < 300:
        return "U_LONG_TAIL_REVIEW"
    if valid_negative_count < 20 or corrected_error_rate is None:
        return "U_INSUFFICIENT_VALID_NEGATIVES"
    if positive_rate is None:
        return "U_INCOMPLETE_POSITIVE_RESULTS"
    if positive_rate >= 0.70 and corrected_error_rate <= 0.10:
        return "A_STABLE_CANDIDATE"
    if positive_rate < 0.55 and corrected_error_rate <= 0.10:
        return "C_NARROW_OR_LEGACY_NOISE_REVIEW"
    if positive_rate >= 0.70 and corrected_error_rate > 0.15:
        return "D_BROAD_BOUNDARY_REVIEW"
    if positive_rate < 0.55 and corrected_error_rate > 0.15:
        return "E_BOUNDARY_CONFLICT_REVIEW"
    return "B_MINOR_BOUNDARY_REVIEW"


def combine_corrected_boundary_assessments(
    positive_per_label_path: str | Path,
    original_negative_per_label_path: str | Path,
    colabel_per_target_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Remove invalid sibling negatives and recompute true boundary errors."""
    positive = {
        str(row["label_id"]): row for row in _read_jsonl(positive_per_label_path)
    }
    negative = {
        str(row["label_id"]): row
        for row in _read_jsonl(original_negative_per_label_path)
    }
    colabel = {
        str(row["target_label_id"]): row for row in _read_jsonl(colabel_per_target_path)
    }
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label_id, pos in sorted(positive.items()):
        neg = negative.get(label_id, {})
        co = colabel.get(label_id, {})
        positive_count = int(pos.get("planned") or pos.get("completed") or 0)
        positive_rate_value = pos.get("match_rate")
        positive_rate = (
            float(positive_rate_value) if positive_rate_value is not None else None
        )
        hard_total = int(neg.get("hard_negative_total") or 0)
        first_accepted = int(neg.get("false_accept") or 0)
        decision_counts = Counter(co.get("decision_counts") or {})
        judged = sum(decision_counts.values())
        if judged != first_accepted:
            raise ValueError(
                f"co-label judgments differ from first-stage acceptances for {label_id}: "
                f"{judged} != {first_accepted}"
            )
        reasonable = decision_counts["合理共标"]
        source_insufficient = decision_counts["来源Label不足以描述该题"]
        true_errors = decision_counts["目标Label边界过宽"]
        unresolved = decision_counts["无法判断"]
        first_rejected = hard_total - first_accepted
        invalid_negatives = reasonable + source_insufficient
        valid_negative_count = first_rejected + true_errors
        corrected_rate = (
            round(true_errors / valid_negative_count, 6)
            if valid_negative_count
            else None
        )
        rows.append(
            {
                "label_id": label_id,
                "label_name": pos.get("label_name", ""),
                "positive_count": positive_count,
                "positive_match_rate": positive_rate,
                "hard_negative_total": hard_total,
                "first_stage_accepted": first_accepted,
                "first_stage_rejected": first_rejected,
                "colabel_decision_counts": dict(sorted(decision_counts.items())),
                "invalid_sibling_negatives": invalid_negatives,
                "unresolved": unresolved,
                "valid_negative_count": valid_negative_count,
                "true_boundary_errors": true_errors,
                "corrected_boundary_error_rate": corrected_rate,
                "final_screen": _corrected_screen(
                    positive_count,
                    positive_rate,
                    valid_negative_count,
                    corrected_rate,
                ),
                "decision_warning": "Corrected errors count only target-boundary-too-broad decisions; reasonable co-label and source-insufficient cases are removed as invalid sibling negatives.",
            }
        )
    with (output_dir / "label_assessments.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "labels": len(rows),
        "positive_labels": len(positive),
        "original_negative_labels": len(negative),
        "colabel_labels": len(colabel),
        "hard_negative_total": sum(row["hard_negative_total"] for row in rows),
        "first_stage_accepted": sum(row["first_stage_accepted"] for row in rows),
        "invalid_sibling_negatives": sum(
            row["invalid_sibling_negatives"] for row in rows
        ),
        "unresolved": sum(row["unresolved"] for row in rows),
        "valid_negative_count": sum(row["valid_negative_count"] for row in rows),
        "true_boundary_errors": sum(row["true_boundary_errors"] for row in rows),
        "corrected_boundary_error_rate": None,
        "screen_counts": dict(
            sorted(Counter(row["final_screen"] for row in rows).items())
        ),
        "warning": "Final screens remain evidence-based review priorities, not authorization to rewrite teacher definitions.",
    }
    if report["valid_negative_count"]:
        report["corrected_boundary_error_rate"] = round(
            report["true_boundary_errors"] / report["valid_negative_count"], 6
        )
    _write_json_atomic(output_dir / "report.json", report)
    return report
