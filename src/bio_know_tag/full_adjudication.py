"""Bounded-memory candidate adjudication directly over full aligned JSONL inputs."""

from __future__ import annotations

import itertools
import json
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from bio_know_tag.adjudication import (
    PARENT_PROMPT_VERSION,
    PROMPT_VERSION,
    _ensure_run_manifest,
    _file_sha256,
    _prompt_version_for_unit,
    _write_json_atomic,
    build_adjudication_prompt,
    validate_adjudication_result,
)
from bio_know_tag.ds import DSRequestError, parse_json_content
from bio_know_tag.retrieval import format_label_path


def _rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            yield value


def _pairs(units_path: str | Path, candidates_path: str | Path) -> Iterator[tuple[dict, dict]]:
    for number, (unit, candidate) in enumerate(
        itertools.zip_longest(_rows(units_path), _rows(candidates_path)), 1
    ):
        if unit is None or candidate is None:
            raise ValueError(f"unit/candidate row counts differ at row {number}")
        question_id = str(unit.get("question_id") or "")
        if not question_id or question_id != str(candidate.get("question_id") or ""):
            raise ValueError(f"question_id mismatch at row {number}")
        yield unit, candidate


def _record_request(unit: dict, candidate: dict, labels: dict, client: Any, model: str, max_tokens: int) -> dict:
    question_id = str(unit["question_id"])
    prompt, code_map = build_adjudication_prompt(unit, candidate["candidates"], labels)
    record = {
        "stage": "candidate_adjudication", "question_id": question_id,
        "prompt_version": _prompt_version_for_unit(unit),
        "candidate_code_map": code_map, "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": None, "parsed_response": None,
        "endpoint": None, "attempts": 0, "latency_seconds": None,
        "usage": None, "reasoning": None, "retry_errors": [],
        "finish_reason": None, "stream": True, "error": None,
    }
    try:
        response = client.chat(
            [
                {"role": "system", "content": "你是严谨的高中生物知识点判标器，只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stream=True,
        )
        record.update({
            "raw_response": response.content,
            "endpoint": response.endpoint,
            "attempts": response.attempts,
            "latency_seconds": response.latency_seconds,
            "usage": getattr(response, "usage", None),
            "reasoning": getattr(response, "reasoning", None),
            "retry_errors": list(getattr(response, "retry_errors", ())),
            "finish_reason": getattr(response, "finish_reason", None),
        })
        if record["finish_reason"] == "length":
            raise ValueError("completion truncated because finish_reason=length")
        parsed = validate_adjudication_result(parse_json_content(response.content), set(code_map))
        record["parsed_response"] = parsed
    except DSRequestError as exc:
        record.update({
            "endpoint": exc.endpoint, "attempts": exc.attempts,
            "latency_seconds": exc.latency_seconds,
            "retry_errors": list(exc.retry_errors),
            "error": f"{type(exc).__name__}: {exc}",
        })
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _prediction(unit: dict, candidate_row: dict, record: dict, labels: dict, model: str) -> dict:
    parsed = record["parsed_response"]
    candidates = candidate_row.get("candidates") or []
    candidates_by_id = {str(item["label_id"]): item for item in candidates}
    selected = []
    for code in parsed["selected"]:
        label_id = str(record["candidate_code_map"][code])
        candidate = candidates_by_id[label_id]
        label = labels[label_id]
        selected.append({
            "label_id": label_id,
            "label_name": label.get("label_name", ""),
            "label_path": format_label_path(label.get("label_path")),
            "candidate_rank": int(candidate.get("candidate_rank") or candidate.get("rank") or 0),
            "sources": candidate.get("sources", []),
            "sparse_rank": candidate.get("sparse_rank"),
            "dense_rank": candidate.get("dense_rank"),
            "evidence": parsed["evidence"][code],
        })
    selected.sort(key=lambda row: (row["candidate_rank"], row["label_id"]))
    if unit.get("unit_type") == "composite_parent_extra":
        text_missing = not any(str(unit.get(field) or "").strip() for field in ("stem", "options", "answer_text", "analysis"))
    else:
        text_missing = not str(unit.get("stem") or "").strip() and not str(unit.get("parent_stem") or "").strip()
    needs_review = bool(parsed["context_insufficient"] or parsed["need_expand_recall"] or text_missing)
    return {
        "question_id": str(unit["question_id"]),
        "parent_id": unit.get("parent_id", unit["question_id"]),
        "unit_type": unit.get("unit_type", ""),
        "reason": parsed["reason"],
        "selected_labels": selected,
        "audited_excluded_labels": [],
        "unknown_selected_codes_dropped": parsed.get("unknown_selected_codes_dropped", []),
        "none_of_candidates": parsed["none_of_candidates"],
        "need_expand_recall": parsed["need_expand_recall"],
        "context_insufficient": parsed["context_insufficient"],
        "text_content_missing": text_missing,
        "needs_review": needs_review,
        "usable_for_training": bool(selected and not needs_review),
        "candidate_count": len(candidates),
        "retrieval_version": candidate_row.get("retrieval_version", ""),
        "model": model,
        "prompt_version": _prompt_version_for_unit(unit),
    }


def run_full_adjudication(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    client: Any,
    *,
    model: str,
    workers: int = 25,
    max_tokens: int = 1024,
    progress_every: int = 1000,
    request_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """No shard files: stream aligned rows through at most 2×workers futures."""
    if workers < 1 or max_tokens < 1:
        raise ValueError("workers and max_tokens must be positive")
    started = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = {str(row["label_id"]): row for row in _rows(labels_path)}
    manifest = {
        "runner_version": "full-adjudication-bounded-v1",
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "audited_exclusions": None,
        "request_config": request_config,
        "prompt_versions_by_unit_type": {
            "composite_parent_extra": PARENT_PROMPT_VERSION,
            "other": PROMPT_VERSION,
        },
        "input_paths": {"units": str(units_path), "candidates": str(candidates_path), "labels": str(labels_path)},
        "input_sha256": {
            "units": _file_sha256(units_path),
            "candidates": _file_sha256(candidates_path),
            "labels": _file_sha256(labels_path),
        },
    }
    _ensure_run_manifest(output / "run_manifest.json", manifest)
    database = sqlite3.connect(output / "state.sqlite3")
    database.execute("CREATE TABLE IF NOT EXISTS completed (question_id TEXT PRIMARY KEY, evidence_offset INTEGER NOT NULL)")
    database.execute("CREATE TABLE IF NOT EXISTS input_ids (question_id TEXT PRIMARY KEY)")
    evidence_path = output / "evidence.jsonl"
    try:
        # Validate the entire source before sending any request. This is cheap
        # compared with generation and prevents silent cross-file misalignment.
        database.execute("DELETE FROM input_ids")
        database.commit()
        input_count = 0
        by_type: Counter[str] = Counter()
        for unit, candidate in _pairs(units_path, candidates_path):
            qid = str(unit["question_id"])
            database.execute("INSERT INTO input_ids VALUES (?)", (qid,))
            items = candidate.get("candidates")
            if not isinstance(items, list):
                raise ValueError(f"candidate list missing for {qid}")
            for item in items:
                if str(item.get("label_id") or "") not in labels:
                    raise ValueError(f"unknown candidate label for {qid}")
            input_count += 1
            by_type[str(unit.get("unit_type") or "unknown")] += 1
        database.commit()
        if input_count < 1:
            raise ValueError("no input units")
        evidence_rows = 0
        if evidence_path.exists():
            with evidence_path.open("rb") as evidence:
                while True:
                    offset = evidence.tell()
                    line = evidence.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        raise ValueError("incomplete evidence tail; rerun via six-vote controller for recovery")
                    record = json.loads(line)
                    evidence_rows += 1
                    if not record.get("error") and isinstance(record.get("parsed_response"), dict):
                        if not database.execute(
                            "SELECT 1 FROM input_ids WHERE question_id=?",
                            (str(record["question_id"]),),
                        ).fetchone():
                            raise ValueError(f"successful evidence references unknown question_id: {record['question_id']}")
                        database.execute(
                            "INSERT OR REPLACE INTO completed VALUES (?, ?)",
                            (str(record["question_id"]), offset),
                        )
            database.commit()
        success_before = database.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        attempted = failed = succeeded_this_run = 0

        with ThreadPoolExecutor(max_workers=workers) as executor, evidence_path.open("ab") as evidence:
            pending = set()

            def persist(done_futures: set) -> None:
                nonlocal attempted, failed, succeeded_this_run, evidence_rows
                for future in done_futures:
                    record = future.result()
                    offset = evidence.tell()
                    evidence.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                    evidence.flush()
                    os.fsync(evidence.fileno())
                    evidence_rows += 1
                    attempted += 1
                    failed += int(bool(record["error"]))
                    if not record["error"]:
                        succeeded_this_run += 1
                        database.execute(
                            "INSERT OR REPLACE INTO completed VALUES (?, ?)",
                            (str(record["question_id"]), offset),
                        )
                    if progress_every and attempted % progress_every == 0:
                        database.commit()
                        _write_json_atomic(output / "report.json", {
                            "input": input_count,
                            "success": success_before + succeeded_this_run,
                            "processed": success_before + succeeded_this_run,
                            "error": input_count - success_before - succeeded_this_run,
                            "pending": input_count - success_before - succeeded_this_run,
                            "requests_failed_this_run": failed,
                            "requests_succeeded_this_run": succeeded_this_run,
                            "evidence_rows": evidence_rows,
                            "workers": workers,
                            "stream_transport_this_run": True,
                            "model": model,
                            "status": "running",
                        })
                        print(
                            f"full adjudication: attempts={attempted} errors={failed} "
                            f"input={input_count}",
                            flush=True,
                        )

            for unit, candidate in _pairs(units_path, candidates_path):
                qid = str(unit["question_id"])
                if database.execute("SELECT 1 FROM completed WHERE question_id=?", (qid,)).fetchone():
                    continue
                pending.add(executor.submit(_record_request, unit, candidate, labels, client, model, max_tokens))
                if len(pending) >= 2 * workers:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    persist(finished)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                persist(finished)
            database.commit()

        predictions = output / "predictions.jsonl"
        temporary = predictions.with_name(f".{predictions.name}.tmp")
        selected_count: Counter[str] = Counter()
        written = 0
        with evidence_path.open("rb") as evidence, temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for unit, candidate in _pairs(units_path, candidates_path):
                result = database.execute(
                    "SELECT evidence_offset FROM completed WHERE question_id=?",
                    (str(unit["question_id"]),),
                ).fetchone()
                if result is None:
                    continue
                evidence.seek(int(result[0]))
                record = json.loads(evidence.readline())
                if record.get("prompt_version") != _prompt_version_for_unit(unit):
                    raise ValueError(f"prompt version mismatch for {unit['question_id']}")
                prediction = _prediction(unit, candidate, record, labels, model)
                selected_count[str(len(prediction["selected_labels"]))] += 1
                destination.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
        temporary.replace(predictions)
        report = {
            "input": input_count,
            "success": written,
            "processed": written,
            "error": input_count - written,
            "pending": input_count - written,
            "requests_succeeded_this_run": succeeded_this_run,
            "requests_failed_this_run": failed,
            "evidence_rows": evidence_rows,
            "workers": workers,
            "max_pending_futures": 2 * workers,
            "stream_transport_this_run": True,
            "model": model,
            "max_tokens": max_tokens,
            "unit_types": dict(by_type),
            "selected_count_distribution": dict(selected_count),
            "run_wall_seconds": round(time.monotonic() - started, 3),
            "input_sha256": manifest["input_sha256"],
        }
        _write_json_atomic(output / "report.json", report)
        return report
    finally:
        database.close()
