"""Read-only timing estimates for an in-progress six-vote adjudication run."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


VOTES = ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3")


def _recent_complete_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read only the final complete JSONL records, not a multi-GB evidence file."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("rb") as handle:
        position = handle.seek(0, 2)
        chunks: list[bytes] = []
        line_breaks = 0
        while position > 0 and line_breaks <= limit:
            size = min(position, 64 * 1024)
            position -= size
            handle.seek(position)
            chunk = handle.read(size)
            chunks.append(chunk)
            line_breaks += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    lines = data.split(b"\n")
    if position > 0:
        lines = lines[1:]  # The first buffered line may be incomplete.
    if lines and not lines[-1]:
        lines.pop()
    else:
        lines = lines[:-1]  # Ignore an unfinished append-only record.
    rows = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _read_report(path: Path, total: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report.get("input", -1)) != total:
        raise ValueError(f"{path} input count differs from --total={total}")
    return report


def estimate_full_votes(
    run_dir: str | Path,
    *,
    total: int,
    now: datetime | None = None,
    window: int = 1000,
) -> dict[str, Any]:
    if total < 1 or window < 2:
        raise ValueError("total must be positive and window must be at least 2")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must include a timezone")
    root = Path(run_dir)
    votes = []
    for name in VOTES:
        directory = root / "votes" / name
        report_path = directory / "report.json"
        report = _read_report(report_path, total)
        evidence_path = directory / "evidence.jsonl"
        records = _recent_complete_rows(evidence_path, window)
        successful_times = []
        for record in records:
            if record.get("error") or not isinstance(record.get("parsed_response"), dict):
                continue
            try:
                stamp = datetime.fromisoformat(str(record["created_at"]).replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError):
                continue
            if stamp.tzinfo is not None:
                successful_times.append(stamp)
        duration = (
            (max(successful_times) - min(successful_times)).total_seconds()
            if len(successful_times) >= 2 else 0.0
        )
        rate = round((len(successful_times) - 1) / duration, 6) if duration > 0 else None
        success = int(report["success"]) if report else 0
        if success < 0 or success > total:
            raise ValueError(f"invalid success count in {report_path}")
        remaining = total - success
        eta = round(remaining / rate, 1) if rate and len(successful_times) >= 20 else None
        phase = (
            "complete" if remaining == 0 else
            "running" if report and eta is not None else
            "warming_up" if records else
            "waiting_for_results"
        )
        votes.append({
            "vote": name,
            "phase": phase,
            "success_at_last_report": success if report else None,
            "progress_percent": round(100 * success / total, 2) if report else None,
            "recent_complete_records": len(records),
            "recent_successes": len(successful_times),
            "recent_successes_per_second": rate,
            "latest_request_at": max(successful_times).isoformat() if successful_times else None,
            "estimated_remaining_seconds": eta if remaining else 0.0,
            "report_age_seconds": round(current.timestamp() - report_path.stat().st_mtime, 1) if report else None,
            "evidence_age_seconds": round(current.timestamp() - evidence_path.stat().st_mtime, 1) if records else None,
        })
    estimable = all(vote["estimated_remaining_seconds"] is not None for vote in votes)
    slowest = max(votes, key=lambda vote: vote["estimated_remaining_seconds"]) if estimable else None
    remaining_seconds = slowest["estimated_remaining_seconds"] if slowest else None
    return {
        "run_dir": str(Path(run_dir)),
        "total_per_vote": total,
        "as_of": current.isoformat(),
        "votes": votes,
        "slowest_vote": slowest["vote"] if slowest else None,
        "estimated_remaining_seconds": remaining_seconds,
        "estimated_finish_at": (current + timedelta(seconds=remaining_seconds)).isoformat() if slowest else None,
        "estimate_note": "Recent speed uses request-created timestamps in completed evidence; it is approximate, especially during startup, stalls, or restarts.",
    }
