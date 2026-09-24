"""Run Qwen and DS three times each over one unsharded full input."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bio_know_tag.six_vote_runner import (
    DS_VOTES,
    QWEN_VOTES,
    _interrupt_on_sigterm,
    _preserve_incomplete_evidence_tail,
    _write_json_atomic,
)
from bio_know_tag.three_vote_runner import _model_preflight, _sha256


def _vote_command(
    units: Path,
    candidates: Path,
    labels: Path,
    vote_dir: Path,
    service: dict[str, Any],
    workers: int,
    retry_delay: float,
) -> list[str]:
    return [
        sys.executable, "scripts/run_candidate_adjudication_full.py",
        "--units", str(units),
        "--candidates", str(candidates),
        "--labels", str(labels),
        "--run-dir", str(vote_dir),
        "--endpoint", service["endpoint"],
        "--model", service["model"],
        "--workers", str(workers),
        "--max-tokens", str(service["max_tokens"]),
        "--timeout", str(service["timeout"]),
        "--retries", str(service["retries"]),
        "--retry-delay", str(retry_delay),
    ]


def run_full_six_votes(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    *,
    qwen_endpoint: str = "http://172.22.0.35:9204/v1/chat/completions",
    qwen_model: str = "qwen3.8-27b-fp8",
    ds_endpoint: str = "http://172.22.0.35:9205/v1/chat/completions",
    ds_model: str = "ds-v4-flash",
    workers_per_vote: int = 25,
    max_tokens: int = 1024,
    qwen_timeout: float = 600,
    ds_timeout: float = 300,
    retries: int = 5,
    retry_delay: float = 1,
    preflight: bool = True,
) -> dict[str, Any]:
    if min(workers_per_vote, max_tokens, retries) < 1:
        raise ValueError("workers, max_tokens and retries must be positive")
    if retry_delay < 0 or min(qwen_timeout, ds_timeout) <= 0:
        raise ValueError("timeouts must be positive and retry_delay non-negative")
    if qwen_endpoint == ds_endpoint:
        raise ValueError("Qwen and DS endpoints must differ")
    units = Path(units_path).resolve()
    candidates = Path(candidates_path).resolve()
    labels = Path(labels_path).resolve()
    output = Path(run_dir).resolve()
    for path in (units, candidates, labels):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or empty input: {path}")
    if preflight:
        _model_preflight(qwen_endpoint, qwen_model)
        _model_preflight(ds_endpoint, ds_model)
    services = {
        "qwen": {
            "endpoint": qwen_endpoint, "model": qwen_model,
            "timeout": qwen_timeout, "max_tokens": max_tokens,
            "retries": retries, "votes": list(QWEN_VOTES),
        },
        "ds": {
            "endpoint": ds_endpoint, "model": ds_model,
            "timeout": ds_timeout, "max_tokens": max_tokens,
            "retries": retries, "votes": list(DS_VOTES),
        },
    }
    manifest = {
        "runner_version": "qwen-ds-six-vote-full-stream-v1",
        "input_paths": {"units": str(units), "candidates": str(candidates), "labels": str(labels)},
        "input_sha256": {
            "units": _sha256(units),
            "candidates": _sha256(candidates),
            "labels": _sha256(labels),
        },
        "services": services,
        "workers_per_vote": workers_per_vote,
        "max_client_in_flight_per_service": 3 * workers_per_vote,
        "max_client_in_flight_total": 6 * workers_per_vote,
        "retry_delay": retry_delay,
        "temperature": 0,
        "stream": True,
        "audited_exclusions_applied": False,
        "thinking_override": None,
        "prefix_caching": "server_side_required_not_set_by_client",
        "no_shards": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        with (output / "controller.lock").open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another full six-vote controller is running") from exc
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                    raise ValueError("full six-vote manifest mismatch on resume")
            else:
                _write_json_atomic(manifest_path, manifest)

            processes: list[tuple[str, subprocess.Popen]] = []
            logs = []
            try:
                for service in services.values():
                    for vote in service["votes"]:
                        vote_dir = output / "votes" / vote
                        vote_dir.mkdir(parents=True, exist_ok=True)
                        repaired = _preserve_incomplete_evidence_tail(vote_dir)
                        if repaired:
                            print(f"{vote}: preserved incomplete evidence tail ({repaired} bytes)", flush=True)
                        log = (vote_dir / "runner.log").open("a", encoding="utf-8")
                        logs.append(log)
                        command = _vote_command(
                            units, candidates, labels, vote_dir,
                            service, workers_per_vote, retry_delay,
                        )
                        print(f"start {vote}: {workers_per_vote} workers", flush=True)
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
                            raise RuntimeError(
                                f"incomplete vote {vote}: exit={code}; "
                                "rerun unchanged command to resume"
                            )
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
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    return {
        "votes_complete": list(QWEN_VOTES + DS_VOTES),
        "max_client_in_flight_per_service": 3 * workers_per_vote,
        "max_client_in_flight_total": 6 * workers_per_vote,
        "manifest": str(manifest_path),
    }
