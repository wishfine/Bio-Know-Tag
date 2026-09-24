import json
from pathlib import Path

import pytest

from bio_know_tag.full_six_vote_runner import run_full_six_votes


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    labels = tmp_path / "labels.jsonl"
    units.write_text('{"question_id":"q1"}\n', encoding="utf-8")
    candidates.write_text('{"question_id":"q1","candidates":[]}\n', encoding="utf-8")
    labels.write_text('{"label_id":"L1"}\n', encoding="utf-8")
    return units, candidates, labels


def test_full_six_votes_launch_together_on_same_full_files(tmp_path: Path, monkeypatch):
    units, candidates, labels = _inputs(tmp_path)
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
        def poll(self):
            assert len(commands) == 6
            return 0
        def wait(self, timeout=None):
            assert len(commands) == 6
            return 0
    monkeypatch.setattr("bio_know_tag.full_six_vote_runner.subprocess.Popen", Process)
    report = run_full_six_votes(units, candidates, labels, tmp_path / "output", preflight=False)
    assert report["max_client_in_flight_per_service"] == 75
    assert report["max_client_in_flight_total"] == 150
    assert len({cmd[cmd.index("--units") + 1] for cmd in commands}) == 1
    assert len({cmd[cmd.index("--candidates") + 1] for cmd in commands}) == 1
    assert len({cmd[cmd.index("--run-dir") + 1] for cmd in commands}) == 6
    for cmd in commands:
        assert cmd[1] == "scripts/run_candidate_adjudication_full.py"
        assert cmd[cmd.index("--workers") + 1] == "25"
        assert cmd[cmd.index("--max-tokens") + 1] == "1024"
        assert cmd[cmd.index("--retries") + 1] == "5"
    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text())
    assert manifest["temperature"] == 0
    assert manifest["stream"] is True


def test_full_six_votes_refuses_changed_model_on_resume(tmp_path: Path, monkeypatch):
    units, candidates, labels = _inputs(tmp_path)
    monkeypatch.setattr("bio_know_tag.full_six_vote_runner.subprocess.Popen", lambda *a, **k: type(
        "Process", (), {"poll": lambda self: 0, "wait": lambda self, timeout=None: 0}
    )())
    run_full_six_votes(units, candidates, labels, tmp_path / "output", preflight=False)
    with pytest.raises(ValueError, match="manifest mismatch"):
        run_full_six_votes(units, candidates, labels, tmp_path / "output", preflight=False, ds_model="changed")


def test_full_six_votes_stops_others_on_first_failure(tmp_path: Path, monkeypatch):
    units, candidates, labels = _inputs(tmp_path)
    processes = []

    class Process:
        def __init__(self, command, **kwargs):
            self.vote = Path(command[command.index("--run-dir") + 1]).name
            self.terminated = False
            processes.append(self)

        def poll(self):
            assert len(processes) == 6
            if self.vote == "ds1":
                return 1
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return self.poll()

    monkeypatch.setattr("bio_know_tag.full_six_vote_runner.subprocess.Popen", Process)
    with pytest.raises(RuntimeError, match="incomplete vote ds1"):
        run_full_six_votes(units, candidates, labels, tmp_path / "output", preflight=False)
    assert all(process.terminated for process in processes if process.vote != "ds1")
