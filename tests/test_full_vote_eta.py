import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bio_know_tag.full_vote_eta import estimate_full_votes


VOTES = ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3")


def _write_vote(root: Path, vote: str, *, success: int, interval: int, start: datetime) -> None:
    directory = root / "votes" / vote
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(
        json.dumps({"input": 100, "success": success, "status": "running"}),
        encoding="utf-8",
    )
    with (directory / "evidence.jsonl").open("w", encoding="utf-8") as output:
        for index in range(21):
            output.write(json.dumps({
                "created_at": (start + timedelta(seconds=index * interval)).isoformat(),
                "error": None,
                "parsed_response": {"selected": []},
            }) + "\n")


def test_full_vote_eta_reports_preflight_without_results(tmp_path: Path):
    result = estimate_full_votes(tmp_path, total=100)
    assert len(result["votes"]) == 6
    assert all(vote["phase"] == "waiting_for_results" for vote in result["votes"])
    assert result["estimated_finish_at"] is None


def test_full_vote_eta_uses_slowest_of_six_votes(tmp_path: Path):
    start = datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)
    now = start + timedelta(minutes=10)
    for vote in VOTES:
        _write_vote(tmp_path, vote, success=40, interval=2 if vote == "ds3" else 1, start=start)
    result = estimate_full_votes(tmp_path, total=100, now=now)
    assert result["estimated_remaining_seconds"] == 120.0
    assert result["slowest_vote"] == "ds3"
    assert result["estimated_finish_at"] == (now + timedelta(seconds=120)).isoformat()
    assert all(vote["progress_percent"] == 40.0 for vote in result["votes"])


def test_full_vote_eta_ignores_incomplete_evidence_tail(tmp_path: Path):
    start = datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)
    _write_vote(tmp_path, "qwen1", success=40, interval=1, start=start)
    with (tmp_path / "votes" / "qwen1" / "evidence.jsonl").open("ab") as output:
        output.write(b'{"created_at":')
    result = estimate_full_votes(tmp_path, total=100, now=start + timedelta(minutes=10))
    assert result["votes"][0]["recent_successes"] == 21
    assert result["votes"][0]["recent_successes_per_second"] == 1.0


def test_full_vote_eta_reads_only_recent_large_file_records(tmp_path: Path):
    directory = tmp_path / "votes" / "qwen1"
    directory.mkdir(parents=True)
    start = datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc)
    (directory / "report.json").write_text('{"input":2000,"success":1200}', encoding="utf-8")
    with (directory / "evidence.jsonl").open("w", encoding="utf-8") as output:
        for index in range(1200):
            output.write(json.dumps({
                "created_at": (start + timedelta(seconds=index)).isoformat(),
                "parsed_response": {}, "error": None, "padding": "x" * 200,
            }) + "\n")
    result = estimate_full_votes(tmp_path, total=2000, now=start + timedelta(hours=1))
    vote = result["votes"][0]
    assert vote["recent_complete_records"] == 1000
    assert vote["recent_successes_per_second"] == 1.0
    assert vote["estimated_remaining_seconds"] == 800.0
