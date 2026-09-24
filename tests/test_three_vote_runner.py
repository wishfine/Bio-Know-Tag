import json
from pathlib import Path

import pytest

from bio_know_tag.three_vote_runner import build_vote_command, run_three_vote_shards


def _shard_root(tmp_path: Path) -> Path:
    root = tmp_path / "sharded"
    shard = root / "shards" / "00001"
    shard.mkdir(parents=True)
    (shard / "units.jsonl").write_text('{"question_id":"q1"}\n', encoding="utf-8")
    (shard / "candidates.jsonl").write_text('{"question_id":"q1","candidates":[]}\n', encoding="utf-8")
    (root / "report.json").write_text(json.dumps({"input": 1, "shard_count": 1, "shards": [
        {"shard_id": "00001", "count": 1}
    ]}), encoding="utf-8")
    return root


def test_command_uses_stream_and_qwen_settings(tmp_path: Path):
    root = _shard_root(tmp_path)
    command = build_vote_command(
        root / "shards" / "00001", "run2", "http://host:9204/v1/chat/completions",
        "qwen3.8-27b-fp8", tmp_path / "labels.jsonl", workers=30,
        max_tokens=1024, timeout=600, retries=3, retry_delay=1,
        disable_thinking=True,
    )
    assert command[command.index("--workers") + 1] == "30"
    assert command[command.index("--model") + 1] == "qwen3.8-27b-fp8"
    assert "--stream" in command
    assert "--disable-thinking" in command
    assert "--no-audited-exclusions" in command
    assert command[command.index("--run-dir") + 1].endswith("shards/00001/votes/run2")


def test_three_votes_start_concurrently_and_each_has_own_resume_dir(tmp_path: Path, monkeypatch):
    root = _shard_root(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"label_id":"L1"}\n', encoding="utf-8")
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.returncode = 0
            self.stdout = kwargs["stdout"]
        def wait(self):
            assert len(commands) == 3  # all votes launched before waiting
            self.stdout.close()
            return self.returncode
    monkeypatch.setattr("bio_know_tag.three_vote_runner.subprocess.Popen", Process)
    report = run_three_vote_shards(
        root, labels, "http://host:9204/v1/chat/completions", "qwen3.8-27b-fp8",
        workers_per_vote=30, preflight=False,
    )
    assert report["max_client_in_flight"] == 90
    assert report["shards_launched"] == 1
    assert len({command[command.index("--run-dir") + 1] for command in commands}) == 3


def test_refuses_changed_model_on_resume(tmp_path: Path, monkeypatch):
    root = _shard_root(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{}\n', encoding="utf-8")
    monkeypatch.setattr("bio_know_tag.three_vote_runner.subprocess.Popen", lambda *a, **k: type(
        "Process", (), {"wait": lambda self: 0}
    )())
    run_three_vote_shards(root, labels, "http://host:9204/v1/chat/completions", "model-a", preflight=False)
    with pytest.raises(ValueError, match="manifest mismatch"):
        run_three_vote_shards(root, labels, "http://host:9204/v1/chat/completions", "model-b", preflight=False)
