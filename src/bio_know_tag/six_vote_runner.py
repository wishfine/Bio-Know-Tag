"""Launch three Qwen and three DS votes concurrently for each aligned shard."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_know_tag.three_vote_runner import _model_preflight, _sha256, build_vote_command


QWEN_VOTES = ("qwen1", "qwen2", "qwen3")
DS_VOTES = ("ds1", "ds2", "ds3")


def _preserve_incomplete_evidence_tail(vote_dir: Path) -> int:
    """Keep an interrupted final fragment for audit before removing it from JSONL."""
    evidence = vote_dir / "evidence.jsonl"
    if not evidence.exists():
        return 0
    with evidence.open("rb+") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        if not end:
            return 0
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            return 0
        cursor = end
        boundary = 0
        while cursor:
            width = min(cursor, 8192)
            cursor -= width
            handle.seek(cursor)
            block = handle.read(width)
            last_newline = block.rfind(b"\n")
            if last_newline >= 0:
                boundary = cursor + last_newline + 1
                break
        handle.seek(boundary)
        fragment = handle.read()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        sidecar = vote_dir / f"evidence.incomplete-tail.{timestamp}.{os.getpid()}.bin"
        sidecar.write_bytes(fragment)
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())
        return len(fragment)


def _interrupt_on_sigterm(_signum: int, _frame: Any) -> None:
    # Ignore a second SIGTERM while the exception handler reaps child processes.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt("six-vote controller received SIGTERM")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_six_vote_shards(
    shard_root: str | Path,
    labels_path: str | Path,
    *,
    qwen_endpoint: str = "http://172.22.0.35:9204/v1/chat/completions",
    qwen_model: str = "qwen3.8-27b-fp8",
    ds_endpoint: str = "http://172.22.0.35:9205/v1/chat/completions",
    ds_model: str = "ds-v4-flash",
    workers_per_vote: int = 25,
    qwen_max_tokens: int = 1024,
    ds_max_tokens: int = 512,
    qwen_timeout: float = 600,
    ds_timeout: float = 300,
    qwen_retries: int = 3,
    ds_retries: int = 5,
    retry_delay: float = 1,
    preflight: bool = True,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Keep six vote streams separate and stop on an incomplete shard.

    Each child uses the same shard inputs and the existing streaming adjudicator.
    The three votes for each model run at once, not one model after the other.
    """
    if min(workers_per_vote, qwen_max_tokens, ds_max_tokens, qwen_retries, ds_retries) < 1:
        raise ValueError("workers, token limits and retry counts must be positive")
    if retry_delay < 0 or min(qwen_timeout, ds_timeout) <= 0:
        raise ValueError("timeouts must be positive and retry delay non-negative")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be positive")
    if qwen_endpoint == ds_endpoint:
        raise ValueError("Qwen and DS must use distinct service endpoints")
    root = Path(shard_root).resolve()
    labels = Path(labels_path).resolve()
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("shards"):
        raise ValueError("shard report has no shards")
    if preflight:
        _model_preflight(qwen_endpoint, qwen_model)
        _model_preflight(ds_endpoint, ds_model)
    services = {
        "qwen": {
            "endpoint": qwen_endpoint,
            "model": qwen_model,
            "max_tokens": qwen_max_tokens,
            "timeout": qwen_timeout,
            "retries": qwen_retries,
            "votes": list(QWEN_VOTES),
        },
        "ds": {
            "endpoint": ds_endpoint,
            "model": ds_model,
            "max_tokens": ds_max_tokens,
            "timeout": ds_timeout,
            "retries": ds_retries,
            "votes": list(DS_VOTES),
        },
    }
    manifest = {
        "runner_version": "qwen-ds-six-vote-stream-v1",
        "shard_report_sha256": _sha256(report_path),
        "labels_sha256": _sha256(labels),
        "services": services,
        "workers_per_vote": workers_per_vote,
        "max_client_in_flight_per_service": 3 * workers_per_vote,
        "max_client_in_flight_total": 6 * workers_per_vote,
        "retry_delay": retry_delay,
        "temperature": 0,
        "stream": True,
        "thinking_override": None,
        "audited_exclusions_applied": False,
        "prefix_caching": "must_be_enabled_on_each_service;not_set_by_client",
    }
    manifest_path = root / "six_vote_manifest.json"
    lock_path = root / "six_vote.lock"
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another six-vote controller is already running") from exc
            lock.seek(0)
            lock.truncate()
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing != manifest:
                    raise ValueError("six-vote manifest mismatch on resume")
            else:
                _write_json_atomic(manifest_path, manifest)

            completed = 0
            for shard in report["shards"][:max_shards]:
                _run_one_shard(
                    root, shard, labels, services, workers_per_vote, retry_delay,
                    completed + 1,
                )
                completed += 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    return {
        "shards_completed_this_run": completed,
        "max_client_in_flight_per_service": 3 * workers_per_vote,
        "max_client_in_flight_total": 6 * workers_per_vote,
        "manifest": str(manifest_path),
    }


def _run_one_shard(
    root: Path,
    shard: dict[str, Any],
    labels: Path,
    services: dict[str, dict[str, Any]],
    workers_per_vote: int,
    retry_delay: float,
    completed_this_run: int,
) -> None:
    shard_id = str(shard["shard_id"])
    shard_dir = root / "shards" / shard_id
    for name in ("units.jsonl", "candidates.jsonl"):
        if not (shard_dir / name).is_file():
            raise ValueError(f"missing {name} in shard {shard_id}")
    processes: list[tuple[str, subprocess.Popen]] = []
    logs = []
    try:
        for config in services.values():
            for vote_name in config["votes"]:
                vote_dir = shard_dir / "votes" / vote_name
                vote_dir.mkdir(parents=True, exist_ok=True)
                repaired_bytes = _preserve_incomplete_evidence_tail(vote_dir)
                if repaired_bytes:
                    print(
                        f"shard={shard_id} vote={vote_name} "
                        f"preserved incomplete evidence tail bytes={repaired_bytes}",
                        flush=True,
                    )
                log = (vote_dir / "runner.log").open("a", encoding="utf-8")
                logs.append(log)
                command = build_vote_command(
                    shard_dir, vote_name, config["endpoint"], config["model"], labels,
                    workers=workers_per_vote, max_tokens=config["max_tokens"],
                    timeout=config["timeout"], retries=config["retries"],
                    retry_delay=retry_delay, disable_thinking=False,
                )
                print(f"shard={shard_id} vote={vote_name} start workers={workers_per_vote}", flush=True)
                processes.append((vote_name, subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONUNBUFFERED": "1"},
                )))
        failed = []
        for vote_name, process in processes:
            exit_code = process.wait()
            if exit_code:
                failed.append(f"{vote_name}: exit={exit_code}")
        if failed:
            raise RuntimeError(
                f"shard {shard_id} incomplete ({', '.join(failed)}); "
                "rerun unchanged command to resume successful questions"
            )
        _write_json_atomic(root / "six_vote_progress.json", {
            "shards_completed_this_run": completed_this_run,
            "last_completed_shard": shard_id,
            "votes_per_shard": list(QWEN_VOTES + DS_VOTES),
        })
        print(f"shard={shard_id} all six votes complete", flush=True)
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
