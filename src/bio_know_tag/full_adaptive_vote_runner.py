"""Run two Qwen and two DS votes, then a third vote only on disagreements."""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from bio_know_tag.adjudication import PARENT_PROMPT_VERSION, PROMPT_VERSION
from bio_know_tag.full_six_vote_runner import _vote_command
from bio_know_tag.six_vote_runner import (
    _interrupt_on_sigterm,
    _preserve_incomplete_evidence_tail,
    _write_json_atomic,
)
from bio_know_tag.three_vote_runner import _model_preflight, _sha256


FIRST_VOTES = ("qwen1", "qwen2", "ds1", "ds2")
THIRD_VOTES = {"qwen": "qwen3_disagreement", "ds": "ds3_disagreement"}
STRATEGY_VERSION = "adaptive-q2d2-disagreement-third-v1"


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number} must be a JSON object")
                yield row


def _signature(row: dict[str, Any]) -> tuple[frozenset[str], bool, bool]:
    selected = row.get("selected_labels")
    if not isinstance(selected, list):
        raise ValueError(f"prediction missing selected_labels: {row.get('question_id')}")
    ids = [str(item.get("label_id") or "") for item in selected]
    if any(not label_id for label_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"invalid selected label IDs: {row.get('question_id')}")
    return (
        frozenset(ids),
        bool(row.get("context_insufficient")),
        bool(row.get("need_expand_recall")),
    )


def build_disagreement_inputs(
    units_path: str | Path,
    candidates_path: str | Path,
    prediction_paths: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize only disagreements; frozen source SHA makes this resumable."""
    units = Path(units_path)
    candidates = Path(candidates_path)
    paths = {vote: Path(prediction_paths[vote]) for vote in FIRST_VOTES}
    output = Path(output_dir)
    source_hashes = {
        "units": _sha256(units),
        "candidates": _sha256(candidates),
        **{vote: _sha256(path) for vote, path in paths.items()},
    }
    report_path = output / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("source_sha256") != source_hashes:
            raise ValueError("source predictions changed after disagreement inputs were frozen")
        for model in ("qwen", "ds"):
            for name in ("units", "candidates"):
                target = output / model / f"{name}.jsonl"
                if not target.is_file() or _sha256(target) != report[model]["input_sha256"][name]:
                    raise ValueError(f"frozen disagreement input changed: {target}")
        return report

    for model in ("qwen", "ds"):
        (output / model).mkdir(parents=True, exist_ok=True)
    temporary = {
        (model, name): output / model / f".{name}.jsonl.tmp"
        for model in ("qwen", "ds") for name in ("units", "candidates")
    }
    handles = {key: path.open("w", encoding="utf-8", newline="\n") for key, path in temporary.items()}
    counts = {
        model: {"disagreements": 0, "label_set_disagreements": 0, "review_flag_only_disagreements": 0}
        for model in ("qwen", "ds")
    }
    total = 0
    try:
        iterators = [_rows(units), _rows(candidates), *(_rows(paths[vote]) for vote in FIRST_VOTES)]
        for line_number, rows in enumerate(itertools.zip_longest(*iterators), 1):
            if any(row is None for row in rows):
                raise ValueError(f"input or prediction row counts differ at row {line_number}")
            unit, candidate, qwen1, qwen2, ds1, ds2 = rows
            qid = str(unit.get("question_id") or "")
            if not qid or any(str(row.get("question_id") or "") != qid for row in rows[1:]):
                raise ValueError(f"question_id mismatch at row {line_number}")
            total += 1
            for model, left, right in (("qwen", qwen1, qwen2), ("ds", ds1, ds2)):
                left_signature, right_signature = _signature(left), _signature(right)
                if left_signature == right_signature:
                    continue
                counts[model]["disagreements"] += 1
                if left_signature[0] != right_signature[0]:
                    counts[model]["label_set_disagreements"] += 1
                else:
                    counts[model]["review_flag_only_disagreements"] += 1
                handles[(model, "units")].write(json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n")
                handles[(model, "candidates")].write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
            if total % 250000 == 0:
                print(f"disagreement scan: {total:,} questions", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    report: dict[str, Any] = {
        "strategy_version": STRATEGY_VERSION,
        "input": total,
        "source_sha256": source_hashes,
    }
    for model in ("qwen", "ds"):
        report[model] = dict(counts[model])
        report[model]["input_sha256"] = {}
        for name in ("units", "candidates"):
            target = output / model / f"{name}.jsonl"
            temporary[(model, name)].replace(target)
            report[model]["input_sha256"][name] = _sha256(target)
    _write_json_atomic(report_path, report)
    return report


def _count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def aggregate_adaptive_model(model: str, output: Path) -> dict[str, int]:
    """Keep actual vote counts (2 or 3); never fabricate an unrequested vote."""
    first = _rows(output / "votes" / f"{model}1" / "predictions.jsonl")
    second = _rows(output / "votes" / f"{model}2" / "predictions.jsonl")
    third_path = output / "votes" / THIRD_VOTES[model] / "predictions.jsonl"
    third = _rows(third_path) if third_path.is_file() else iter(())
    destination = output / "model_consensus"
    destination.mkdir(exist_ok=True)
    temporary = destination / f".{model}.jsonl.tmp"
    counts = {"questions": 0, "two_votes_agreed": 0, "third_vote_used": 0, "needs_review": 0}
    with temporary.open("w", encoding="utf-8") as handle:
        for left, right in itertools.zip_longest(first, second):
            if left is None or right is None or left["question_id"] != right["question_id"]:
                raise ValueError(f"{model} first-two predictions not aligned")
            qid = str(left["question_id"])
            rows = [left, right]
            if _signature(left) != _signature(right):
                extra = next(third, None)
                if extra is None or str(extra["question_id"]) != qid:
                    raise ValueError(f"{model} conditional third prediction missing or misaligned: {qid}")
                rows.append(extra)
                counts["third_vote_used"] += 1
            else:
                counts["two_votes_agreed"] += 1
            label_cards = {}
            label_votes: dict[str, int] = {}
            for row in rows:
                for card in row["selected_labels"]:
                    lid = str(card["label_id"])
                    label_cards[lid] = card
                    label_votes[lid] = label_votes.get(lid, 0) + 1
            selected = [
                {**label_cards[lid], "supporting_votes": label_votes[lid], "actual_votes": len(rows)}
                for lid in sorted(label_votes) if label_votes[lid] >= 2
            ]
            needs_review = any(row.get("needs_review") or row.get("context_insufficient") or row.get("need_expand_recall") for row in rows)
            counts["questions"] += 1
            counts["needs_review"] += int(needs_review)
            result = {
                "question_id": qid, "parent_id": left.get("parent_id"),
                "unit_type": left.get("unit_type"), "model": left.get("model"),
                "selected_labels": selected, "actual_vote_count": len(rows),
                "third_vote_triggered": len(rows) == 3,
                "label_vote_counts": label_votes,
                "context_insufficient": any(row.get("context_insufficient") for row in rows),
                "need_expand_recall": any(row.get("need_expand_recall") for row in rows),
                "needs_review": bool(needs_review),
                "teacher_calibration_status": "NO_TEACHER_GOLD",
                "usable_for_training": False,
                "strategy_version": STRATEGY_VERSION,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    if next(third, None) is not None:
        raise ValueError(f"{model} extra third-vote predictions not required by first two votes")
    temporary.replace(destination / f"{model}.jsonl")
    return counts


def _vote_complete(
    vote_dir: Path, expected: int, model: str, input_hashes: dict[str, str],
    service: dict[str, Any], retry_delay: float,
) -> bool:
    report_path = vote_dir / "report.json"
    manifest_path = vote_dir / "run_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file() or not (vote_dir / "predictions.jsonl").is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config = {
        "endpoint": service["endpoint"], "timeout": service["timeout"],
        "retries": service["retries"], "retry_delay": retry_delay,
        "request_interval": 0, "enable_thinking": False,
    }
    if (
        manifest.get("model") != model or manifest.get("temperature") != 0
        or manifest.get("stream") is not True
        or manifest.get("max_tokens") != service["max_tokens"]
        or manifest.get("request_config") != expected_config
    ):
        raise ValueError(f"adaptive child request configuration mismatch: {vote_dir.name}")
    return (
        report.get("success") == expected and report.get("error") == 0
        and report.get("model") == model and manifest.get("input_sha256") == input_hashes
        and _count_rows(vote_dir / "predictions.jsonl") == expected
        and manifest.get("prompt_versions_by_unit_type") == {
            "composite_parent_extra": PARENT_PROMPT_VERSION, "other": PROMPT_VERSION,
        }
    )


def _seed_from_six_vote(source: Path, output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Copy only first-two evidence and manifests; never modify the old run."""
    seed_report_path = output / "seed_report.json"
    old_manifest_path = source / "run_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("input_sha256") != manifest["input_sha256"]:
        raise ValueError("old six-vote input SHA256 differs from adaptive inputs")
    if old_manifest.get("thinking_override") != {"qwen": False, "ds": False}:
        raise ValueError("old six-vote run did not disable thinking for both models")
    if old_manifest.get("retry_delay") != manifest["retry_delay"]:
        raise ValueError("old six-vote retry_delay differs from adaptive configuration")
    for model in ("qwen", "ds"):
        old_service = (old_manifest.get("services") or {}).get(model) or {}
        new_service = manifest["services"][model]
        for field in ("endpoint", "model", "max_tokens", "timeout", "retries"):
            if old_service.get(field) != new_service[field]:
                raise ValueError(f"old six-vote {model} {field} differs from adaptive configuration")
    for vote in FIRST_VOTES:
        child = json.loads((source / "votes" / vote / "run_manifest.json").read_text(encoding="utf-8"))
        model = "qwen" if vote.startswith("qwen") else "ds"
        service = manifest["services"][model]
        expected_config = {
            "endpoint": service["endpoint"], "timeout": service["timeout"],
            "retries": service["retries"], "retry_delay": manifest["retry_delay"],
            "request_interval": 0, "enable_thinking": False,
        }
        if (
            child.get("input_sha256") != manifest["input_sha256"]
            or child.get("model") != manifest["services"][model]["model"]
            or child.get("temperature") != 0
            or child.get("stream") is not True
            or child.get("request_config") != expected_config
            or child.get("max_tokens") != service["max_tokens"]
            or child.get("input_paths") != manifest["input_paths"]
            or child.get("prompt_versions_by_unit_type") != {
                "composite_parent_extra": PARENT_PROMPT_VERSION, "other": PROMPT_VERSION,
            }
        ):
            raise ValueError(f"old six-vote child manifest incompatible: {vote}")
    if seed_report_path.exists():
        report = json.loads(seed_report_path.read_text(encoding="utf-8"))
        if report.get("source_run_manifest_sha256") != _sha256(old_manifest_path):
            raise ValueError("old six-vote manifest changed after seeding")
        if any(not (output / "votes" / vote / "evidence.jsonl").is_file() for vote in FIRST_VOTES):
            raise ValueError("seed report exists but a copied vote evidence file is missing")
        return report

    old_lock = source / "controller.lock"
    if not old_lock.is_file():
        raise ValueError("old six-vote controller lock missing; cannot verify it is stopped")
    with old_lock.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("old six-vote controller is still running; stop it before seeding") from exc
        target_votes = output / "votes"
        if target_votes.exists():
            # A process interrupted after atomic publication can finish recording
            # provenance without ever overwriting its copied evidence.
            provenance = target_votes / "seed_provenance.json"
            if not provenance.is_file():
                raise ValueError("adaptive votes directory exists without seed provenance")
            report = json.loads(provenance.read_text(encoding="utf-8"))
            if report.get("source_run_manifest_sha256") != _sha256(old_manifest_path):
                raise ValueError("recovered seed source manifest differs")
            for vote in FIRST_VOTES:
                evidence = target_votes / vote / "evidence.jsonl"
                if not evidence.is_file() or _sha256(evidence) != report["votes"][vote]["evidence_sha256"]:
                    raise ValueError("incomplete adaptive seed publication")
            _write_json_atomic(seed_report_path, report)
            return report
        staging = Path(tempfile.mkdtemp(prefix=".seed-votes-", dir=output))
        copied = {}
        for vote in FIRST_VOTES:
            source_vote = source / "votes" / vote
            target_vote = staging / vote
            target_vote.mkdir(parents=True)
            for name in ("evidence.jsonl", "run_manifest.json"):
                original = source_vote / name
                if not original.is_file():
                    raise ValueError(f"old six-vote {vote} missing {name}")
                shutil.copyfile(original, target_vote / name)
            _preserve_incomplete_evidence_tail(target_vote)
            copied[vote] = {
                "evidence_sha256": _sha256(target_vote / "evidence.jsonl"),
                "evidence_bytes": (target_vote / "evidence.jsonl").stat().st_size,
            }
        report = {
            "source_run_dir": str(source),
            "source_run_manifest_sha256": _sha256(old_manifest_path),
            "votes": copied,
        }
        _write_json_atomic(staging / "seed_provenance.json", report)
        staging.replace(target_votes)
        _write_json_atomic(seed_report_path, report)
        return report


def _run_vote_processes(
    vote_specs: list[tuple[str, Path, Path, Path, dict[str, Any]]],
    *, labels: Path, output: Path, workers: int, retry_delay: float,
) -> None:
    processes: list[tuple[str, subprocess.Popen]] = []
    logs = []
    try:
        for vote, units, candidates, vote_dir, service in vote_specs:
            vote_dir.mkdir(parents=True, exist_ok=True)
            repaired = _preserve_incomplete_evidence_tail(vote_dir)
            if repaired:
                print(f"{vote}: preserved {repaired} bytes of incomplete evidence tail", flush=True)
            log = (vote_dir / "runner.log").open("a", encoding="utf-8")
            logs.append(log)
            command = _vote_command(units, candidates, labels, vote_dir, service, workers, retry_delay)
            print(f"start {vote}: {workers} workers", flush=True)
            processes.append((vote, subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"},
            )))
        remaining = dict(processes)
        while remaining:
            for vote, process in list(remaining.items()):
                code = process.poll()
                if code is None:
                    continue
                remaining.pop(vote)
                if code:
                    raise RuntimeError(f"incomplete adaptive vote {vote}: exit={code}; rerun unchanged command")
            if remaining:
                time.sleep(0.5)
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        for log in logs:
            log.close()


def run_full_adaptive_votes(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    seed_from_six_vote: str | Path | None = None,
    qwen_endpoint: str = "http://172.22.0.35:9204/v1/chat/completions",
    qwen_model: str = "qwen3.8-27b-fp8",
    ds_endpoint: str = "http://172.22.0.35:9205/v1/chat/completions",
    ds_model: str = "ds-v4-flash",
    workers_per_vote: int = 35,
    max_tokens: int = 1024,
    qwen_timeout: float = 600,
    ds_timeout: float = 300,
    retries: int = 5,
    retry_delay: float = 1,
    preflight: bool = True,
) -> dict[str, Any]:
    if min(workers_per_vote, max_tokens, retries) < 1 or min(qwen_timeout, ds_timeout) <= 0 or retry_delay < 0:
        raise ValueError("workers, token limit, retries and timeouts must be positive")
    if qwen_endpoint == ds_endpoint:
        raise ValueError("Qwen and DS endpoints must differ")
    units = Path(units_path).resolve()
    candidates = Path(candidates_path).resolve()
    labels = Path(labels_path).resolve()
    output = Path(run_dir).resolve()
    seed = Path(seed_from_six_vote).resolve() if seed_from_six_vote else None
    for path in (units, candidates, labels):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or empty input: {path}")
    if seed is not None and (seed == output or output in seed.parents or seed in output.parents):
        raise ValueError("new adaptive run directory must be separate from old six-vote directory")
    if preflight:
        _model_preflight(qwen_endpoint, qwen_model)
        _model_preflight(ds_endpoint, ds_model)
    services = {
        "qwen": {"endpoint": qwen_endpoint, "model": qwen_model, "max_tokens": max_tokens, "timeout": qwen_timeout, "retries": retries},
        "ds": {"endpoint": ds_endpoint, "model": ds_model, "max_tokens": max_tokens, "timeout": ds_timeout, "retries": retries},
    }
    input_hashes = {"units": _sha256(units), "candidates": _sha256(candidates), "labels": _sha256(labels)}
    input_count = _count_rows(units)
    manifest = {
        "runner_version": STRATEGY_VERSION,
        "input_paths": {"units": str(units), "candidates": str(candidates), "labels": str(labels)},
        "input_sha256": input_hashes,
        "input_count": input_count,
        "services": services,
        "workers_per_vote": workers_per_vote,
        "retry_delay": retry_delay,
        "temperature": 0,
        "stream": True,
        "thinking_override": {"qwen": False, "ds": False},
        "third_vote_trigger": "selected_label_set_or_context_insufficient_or_need_expand_recall_differs",
        "seed_from_six_vote": str(seed) if seed else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        with (output / "controller.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another adaptive controller is already running") from exc
            manifest_path = output / "run_manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                    raise ValueError("adaptive manifest mismatch on resume")
            else:
                _write_json_atomic(manifest_path, manifest)
            if seed:
                _seed_from_six_vote(seed, output, manifest)
            elif (output / "seed_report.json").exists():
                raise ValueError("seed report exists but --seed-from-six-vote was omitted")

            phase1_specs = []
            for model, votes in (("qwen", FIRST_VOTES[:2]), ("ds", FIRST_VOTES[2:])):
                service = services[model]
                for vote in votes:
                    vote_dir = output / "votes" / vote
                    if _vote_complete(vote_dir, input_count, service["model"], input_hashes, service, retry_delay):
                        print(f"skip completed {vote}", flush=True)
                        continue
                    phase1_specs.append((vote, units, candidates, vote_dir, service))
            _run_vote_processes(phase1_specs, labels=labels, output=output, workers=workers_per_vote, retry_delay=retry_delay)
            for model, votes in (("qwen", FIRST_VOTES[:2]), ("ds", FIRST_VOTES[2:])):
                for vote in votes:
                    if not _vote_complete(output / "votes" / vote, input_count, services[model]["model"], input_hashes, services[model], retry_delay):
                        raise RuntimeError(f"first-two vote incomplete after subprocess: {vote}")

            prediction_paths = {vote: output / "votes" / vote / "predictions.jsonl" for vote in FIRST_VOTES}
            split = build_disagreement_inputs(units, candidates, prediction_paths, output / "disagreements")
            phase3_specs = []
            for model, vote in THIRD_VOTES.items():
                count = split[model]["disagreements"]
                vote_dir = output / "votes" / vote
                if not count:
                    print(f"skip {vote}: no first-two disagreements", flush=True)
                    continue
                subset_units = output / "disagreements" / model / "units.jsonl"
                subset_candidates = output / "disagreements" / model / "candidates.jsonl"
                subset_hashes = {"units": split[model]["input_sha256"]["units"], "candidates": split[model]["input_sha256"]["candidates"], "labels": input_hashes["labels"]}
                if _vote_complete(vote_dir, count, services[model]["model"], subset_hashes, services[model], retry_delay):
                    print(f"skip completed {vote}", flush=True)
                    continue
                phase3_specs.append((vote, subset_units, subset_candidates, vote_dir, services[model]))
            _run_vote_processes(phase3_specs, labels=labels, output=output, workers=workers_per_vote, retry_delay=retry_delay)
            for model, vote in THIRD_VOTES.items():
                count = split[model]["disagreements"]
                if not count:
                    continue
                subset_hashes = {"units": split[model]["input_sha256"]["units"], "candidates": split[model]["input_sha256"]["candidates"], "labels": input_hashes["labels"]}
                if not _vote_complete(output / "votes" / vote, count, services[model]["model"], subset_hashes, services[model], retry_delay):
                    raise RuntimeError(f"conditional third vote incomplete: {vote}")
            consensus = {model: aggregate_adaptive_model(model, output) for model in ("qwen", "ds")}
            result = {
                "status": "complete", "strategy_version": STRATEGY_VERSION,
                "input": input_count, "first_votes": list(FIRST_VOTES),
                "conditional_third_votes": split,
                "model_consensus": consensus,
                "workers_per_vote": workers_per_vote,
                "peak_client_in_flight_per_service": 2 * workers_per_vote,
                "manifest": str(manifest_path),
            }
            _write_json_atomic(output / "report.json", result)
            return result
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
