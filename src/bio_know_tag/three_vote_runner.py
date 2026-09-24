"""Run three independent, resumable Qwen votes over aligned input shards."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import urlopen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_preflight(endpoint: str, model: str) -> None:
    if not endpoint.endswith("/v1/chat/completions"):
        raise ValueError("endpoint must end with /v1/chat/completions")
    models_url = endpoint.removesuffix("/chat/completions") + "/models"
    with urlopen(models_url, timeout=30) as response:
        payload = json.load(response)
    served = {str(item.get("id")) for item in payload.get("data", [])}
    if model not in served:
        raise ValueError(f"model {model!r} not served by {models_url}: {sorted(served)}")


def build_vote_command(
    shard_dir: Path,
    vote_name: str,
    endpoint: str,
    model: str,
    labels_path: Path,
    *,
    workers: int,
    max_tokens: int,
    timeout: float,
    retries: int,
    retry_delay: float,
    disable_thinking: bool,
) -> list[str]:
    command = [
        sys.executable, "scripts/run_candidate_adjudication.py",
        "--units", str(shard_dir / "units.jsonl"),
        "--candidates", str(shard_dir / "candidates.jsonl"),
        "--labels", str(labels_path),
        "--run-dir", str(shard_dir / "votes" / vote_name),
        "--endpoint", endpoint,
        "--model", model,
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--retries", str(retries),
        "--retry-delay", str(retry_delay),
        "--request-interval", "0",
        "--max-tokens", str(max_tokens),
        "--stream",
        "--no-audited-exclusions",
    ]
    if disable_thinking:
        command.append("--disable-thinking")
    return command


def run_three_vote_shards(
    shard_root: str | Path,
    labels_path: str | Path,
    endpoint: str,
    model: str,
    *,
    workers_per_vote: int = 30,
    max_tokens: int = 1024,
    timeout: float = 600,
    retries: int = 3,
    retry_delay: float = 1,
    disable_thinking: bool = False,
    preflight: bool = True,
    max_shards: int | None = None,
) -> dict[str, Any]:
    if workers_per_vote < 1 or max_tokens < 1 or retries < 1:
        raise ValueError("workers, max_tokens and retries must be positive")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be positive")
    root = Path(shard_root)
    labels = Path(labels_path)
    shard_report_path = root / "report.json"
    shard_report = json.loads(shard_report_path.read_text(encoding="utf-8"))
    if not shard_report.get("shards"):
        raise ValueError("shard report has no shards")
    if preflight:
        _model_preflight(endpoint, model)
    manifest = {
        "runner_version": "qwen-three-vote-stream-v1",
        "shard_report_sha256": _sha256(shard_report_path),
        "labels_sha256": _sha256(labels),
        "endpoint": endpoint,
        "model": model,
        "votes": ["run1", "run2", "run3"],
        "workers_per_vote": workers_per_vote,
        "max_client_in_flight": 3 * workers_per_vote,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "retries": retries,
        "retry_delay": retry_delay,
        "temperature": 0,
        "stream": True,
        "disable_thinking": disable_thinking,
        "audited_exclusions_applied": False,
        "prefix_caching": "server_side_required_not_set_by_client",
    }
    manifest_path = root / "three_vote_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("three-vote manifest mismatch on resume")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    launched = 0
    for shard in shard_report["shards"][:max_shards]:
        shard_id = str(shard["shard_id"])
        shard_dir = root / "shards" / shard_id
        for name in ("units.jsonl", "candidates.jsonl"):
            if not (shard_dir / name).is_file():
                raise ValueError(f"missing {name} in shard {shard_id}")
        processes = []
        logs = []
        try:
            for vote_name in manifest["votes"]:
                vote_dir = shard_dir / "votes" / vote_name
                vote_dir.mkdir(parents=True, exist_ok=True)
                log = (vote_dir / "runner.log").open("a", encoding="utf-8")
                logs.append(log)
                command = build_vote_command(
                    shard_dir, vote_name, endpoint, model, labels,
                    workers=workers_per_vote, max_tokens=max_tokens,
                    timeout=timeout, retries=retries, retry_delay=retry_delay,
                    disable_thinking=disable_thinking,
                )
                print(f"shard={shard_id} vote={vote_name} start workers={workers_per_vote}", flush=True)
                processes.append((vote_name, subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ, "PYTHONPATH": "src"},
                )))
            failures = []
            for vote_name, process in processes:
                code = process.wait()
                if code:
                    failures.append(f"{vote_name}: exit={code}")
            if failures:
                raise RuntimeError(f"shard {shard_id} incomplete ({', '.join(failures)}); rerun to resume successful questions")
            launched += 1
            print(f"shard={shard_id} all three votes complete", flush=True)
        finally:
            for log in logs:
                log.close()
    return {"shards_launched": launched, "max_client_in_flight": 3 * workers_per_vote, "manifest": str(manifest_path)}
