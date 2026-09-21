#!/usr/bin/env python3
"""Run a four-condition DeepSeek stability experiment on prior changed questions.

The experiment reuses the production adjudication prompt and candidate cards.  It
does not change production adjudication behavior.  Each condition stores every
raw choice, parsed selection, request error, latency, and usage record so that a
later audit can distinguish service failures from model disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bio_know_tag.adjudication import PROMPT_VERSION
from bio_know_tag.ds import parse_json_content
from bio_know_tag.retrieval import format_label_path
from bio_know_tag.ds_stability import (
    StabilityCondition,
    build_task_payload,
    choose_unstable_rows,
    default_conditions,
    jaccard,
    parse_choice,
    parse_condition,
    strict_majority_ids,
)


SYSTEM_MESSAGE = "你是严谨的高中生物知识点判标器，只输出JSON。"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels(path: str | Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["label_id"]): row
        for row in _read_jsonl(path)
        if row.get("label_id")
    }


def _indexed_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id") or "")
        if question_id:
            result[question_id] = row
    return result


class _EndpointPool:
    def __init__(self, endpoints: list[str], request_interval: float) -> None:
        if not endpoints:
            raise ValueError("at least one endpoint is required")
        self.endpoints = [value.rstrip("/") for value in endpoints]
        self.request_interval = request_interval
        self._index = 0
        self._endpoint_lock = threading.Lock()
        self._slot_lock = threading.Lock()
        self._next_request_time = 0.0

    def endpoint(self) -> str:
        with self._endpoint_lock:
            value = self.endpoints[self._index]
            self._index = (self._index + 1) % len(self.endpoints)
            return value

    def wait_slot(self) -> None:
        if not self.request_interval:
            return
        with self._slot_lock:
            now = time.monotonic()
            start = max(now, self._next_request_time)
            self._next_request_time = start + self.request_interval
        if start > now:
            time.sleep(start - now)


def _request(
    pool: _EndpointPool,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    n: int,
    seed: int | None,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_delay: float,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "n": n,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    retry_errors: list[dict[str, Any]] = []
    started = time.monotonic()
    for attempt in range(1, retries + 1):
        endpoint = pool.endpoint()
        try:
            pool.wait_slot()
            request = Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            choices = response_body.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("response has no choices")
            return {
                "response": response_body,
                "endpoint": endpoint,
                "attempts": attempt,
                "latency_seconds": round(time.monotonic() - started, 3),
                "retry_errors": retry_errors,
                "payload_sha256": hashlib.sha256(body).hexdigest(),
            }
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            retry_errors.append(
                {
                    "attempt": attempt,
                    "endpoint": endpoint,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if attempt < retries and retry_delay:
                time.sleep(retry_delay * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"request failed after {retries} attempts: {retry_errors[-1]['error']}"
    )


def _choice_records(
    response_body: dict[str, Any], code_map: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for index, choice in enumerate(response_body.get("choices") or []):
        message = choice.get("message") if isinstance(choice, dict) else None
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        record: dict[str, Any] = {
            "index": index,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "raw_response": content,
            "parsed_response": None,
            "selected_label_ids": [],
            "parse_error": None,
            "reasoning": message.get("reasoning", message.get("reasoning_content")),
        }
        if not isinstance(content, str) or not content.strip():
            record["parse_error"] = "ValueError: empty choice content"
        else:
            parsed, selected, error = parse_choice(content, code_map)
            record["parsed_response"] = parsed
            record["selected_label_ids"] = sorted(selected)
            record["parse_error"] = error
        output.append(record)
    return output


def _latest_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error"):
                completed[str(row.get("question_id"))] = row
    return completed


def _condition_report(
    condition: StabilityCondition,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successes = [row for row in rows if not row.get("error")]
    choices = [choice for row in successes for choice in row.get("choices", [])]
    parse_errors = Counter(
        str(choice.get("parse_error")).split(":", 1)[0]
        for choice in choices
        if choice.get("parse_error")
    )
    vote_sets = []
    within_exact = 0
    within_jaccard = []
    for row in successes:
        valid = [
            set(choice.get("selected_label_ids") or [])
            for choice in row.get("choices", [])
            if not choice.get("parse_error")
        ]
        if not valid:
            continue
        vote_sets.append(strict_majority_ids(valid))
        within_exact += int(len({tuple(sorted(value)) for value in valid}) == 1)
        if len(valid) > 1:
            pairs = [
                jaccard(valid[i], valid[j])
                for i in range(len(valid))
                for j in range(i + 1, len(valid))
            ]
            within_jaccard.extend(pairs)
    return {
        "name": condition.name,
        "temperature": condition.temperature,
        "n": condition.n,
        "workers": condition.workers,
        "seed": condition.seed,
        "input": len(rows),
        "success": len(successes),
        "error": len(rows) - len(successes),
        "choice_count": len(choices),
        "parse_error_count": sum(parse_errors.values()),
        "parse_error_types": dict(parse_errors),
        "majority_nonempty": sum(bool(value) for value in vote_sets),
        "mean_majority_selected_count": round(
            sum(len(value) for value in vote_sets) / len(vote_sets), 6
        ) if vote_sets else None,
        "within_choice_exact_agreement": round(
            within_exact / len(vote_sets), 6
        ) if vote_sets and condition.n > 1 else None,
        "within_choice_mean_jaccard": round(
            sum(within_jaccard) / len(within_jaccard), 6
        ) if within_jaccard else None,
    }


def _pairwise_report(
    records_by_condition: dict[str, dict[str, dict[str, Any]]],
    conditions: list[StabilityCondition],
    groups_by_question: dict[str, str],
) -> list[dict[str, Any]]:
    reports = []
    for left_index, left in enumerate(conditions):
        for right in conditions[left_index + 1 :]:
            left_rows = records_by_condition.get(left.name, {})
            right_rows = records_by_condition.get(right.name, {})
            common = sorted(set(left_rows) & set(right_rows))
            exact = 0
            jaccards: list[float] = []
            by_group: dict[str, dict[str, float]] = {}
            for question_id in common:
                left_choices = [
                    set(choice.get("selected_label_ids") or [])
                    for choice in left_rows[question_id].get("choices", [])
                    if not choice.get("parse_error")
                ]
                right_choices = [
                    set(choice.get("selected_label_ids") or [])
                    for choice in right_rows[question_id].get("choices", [])
                    if not choice.get("parse_error")
                ]
                left_ids = strict_majority_ids(left_choices)
                right_ids = strict_majority_ids(right_choices)
                exact += int(left_ids == right_ids)
                jaccards.append(jaccard(left_ids, right_ids))
                group = groups_by_question.get(question_id, "unknown")
                bucket = by_group.setdefault(group, {"total": 0, "different": 0})
                bucket["total"] += 1
                bucket["different"] += int(left_ids != right_ids)
            reports.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "common_success": len(common),
                    "exact_agreement": round(exact / len(common), 6) if common else None,
                    "output_difference_rate": round(1 - exact / len(common), 6) if common else None,
                    "mean_selection_jaccard": round(sum(jaccards) / len(jaccards), 6) if jaccards else None,
                    "by_perturbation_group": by_group,
                }
            )
    return reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    conditions = [parse_condition(value) for value in args.condition] if args.condition else list(default_conditions())
    if len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("condition names must be unique")
    labels = _labels(args.labels)
    unstable_rows = choose_unstable_rows(
        args.unstable_group,
        limit=args.limit,
        seed=args.sample_seed,
    )
    group_by_question = {
        str(row["question_id"]): str(row.get("perturbation_group") or "unknown")
        for row in unstable_rows
    }
    question_ids = set(group_by_question)
    units = _indexed_rows(args.units)
    top25 = _indexed_rows(args.top25_candidates)
    legacy = _indexed_rows(args.legacy_candidates)
    tasks: list[dict[str, Any]] = []
    for row in unstable_rows:
        question_id = str(row["question_id"])
        if question_id not in units:
            raise ValueError(f"question {question_id} missing from units")
        candidates = top25 if group_by_question[question_id] == "A_same_candidate_set" else legacy
        if question_id not in candidates:
            raise ValueError(f"question {question_id} missing from candidate file")
        prompt, code_map = build_task_payload(units[question_id], candidates[question_id], labels)
        tasks.append(
            {
                "question_id": question_id,
                "perturbation_group": group_by_question[question_id],
                "prompt": prompt,
                "code_map": code_map,
                "candidate_count": len(candidates[question_id].get("candidates") or []),
            }
        )
    output_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "ds-stability-prior-output-differences-v1",
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "input_count": len(tasks),
        "sample_seed": args.sample_seed,
        "group_files": [str(path) for path in args.unstable_group],
        "conditions": [condition.__dict__ for condition in conditions],
        "input_sha256": {
            "units": _sha256(args.units),
            "top25_candidates": _sha256(args.top25_candidates),
            "legacy_candidates": _sha256(args.legacy_candidates),
            "labels": _sha256(args.labels),
        },
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("run manifest mismatch; use a new run directory")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    records_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    condition_reports = []
    for condition in conditions:
        print(
            f"[condition={condition.name}] input={len(tasks)} temp={condition.temperature} "
            f"n={condition.n} workers={condition.workers} seed={condition.seed}",
            flush=True,
        )
        condition_dir = output_dir / condition.name
        condition_dir.mkdir(exist_ok=True)
        response_path = condition_dir / "responses.jsonl"
        completed = _latest_completed(response_path)
        pool = _EndpointPool(args.endpoint, args.request_interval)
        pending = [task for task in tasks if task["question_id"] not in completed]

        def one(task: dict[str, Any]) -> dict[str, Any]:
            started = time.monotonic()
            record: dict[str, Any] = {
                "question_id": task["question_id"],
                "perturbation_group": task["perturbation_group"],
                "condition": condition.name,
                "temperature": condition.temperature,
                "n": condition.n,
                "workers": condition.workers,
                "seed": condition.seed,
                "candidate_count": task["candidate_count"],
                "candidate_code_map": task["code_map"],
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
                "error": None,
            }
            try:
                result = _request(
                    pool,
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": task["prompt"]},
                    ],
                    temperature=condition.temperature,
                    n=condition.n,
                    seed=condition.seed,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_delay=args.retry_delay,
                    enable_thinking=args.enable_thinking,
                )
                response_body = result["response"]
                record.update(
                    {
                        "endpoint": result["endpoint"],
                        "attempts": result["attempts"],
                        "latency_seconds": result["latency_seconds"],
                        "retry_errors": result["retry_errors"],
                        "payload_sha256": result["payload_sha256"],
                        "usage": response_body.get("usage"),
                        "choices": _choice_records(response_body, task["code_map"]),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "endpoint": None,
                        "attempts": args.retries,
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "retry_errors": [],
                        "choices": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            return record

        with response_path.open("a", encoding="utf-8", newline="\n") as output:
            if pending:
                with ThreadPoolExecutor(max_workers=condition.workers) as executor:
                    futures = [executor.submit(one, task) for task in pending]
                    for index, future in enumerate(as_completed(futures), 1):
                        record = future.result()
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        output.flush()
                        completed[record["question_id"]] = record
                        if index % args.progress_every == 0 or index == len(pending):
                            print(
                                f"[condition={condition.name}] {index}/{len(pending)} "
                                f"errors={sum(bool(item.get('error')) for item in completed.values())}",
                                flush=True,
                            )
        records_by_condition[condition.name] = completed
        report = _condition_report(condition, list(completed.values()))
        (condition_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        condition_reports.append(report)

    report = {
        "experiment": manifest["experiment"],
        "input": len(tasks),
        "conditions": condition_reports,
        "pairwise": _pairwise_report(records_by_condition, conditions, group_by_question),
        "selection_group_counts": dict(Counter(group_by_question.values())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--unstable-group", type=Path, action="append", required=True)
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.jsonl"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--limit", type=int, default=5391)
    parser.add_argument("--sample-seed", type=int, default=20260921)
    parser.add_argument("--condition", action="append", help="name:temperature:n:workers[:seed]")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")
    if args.retries < 1 or args.timeout <= 0 or args.retry_delay < 0 or args.request_interval < 0:
        raise SystemExit("invalid timeout/retry settings")
    report = run(args)
    print(json.dumps({"run_dir": str(args.run_dir), **report}, ensure_ascii=False))
    return 0 if all(item["error"] == 0 for item in report["conditions"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
