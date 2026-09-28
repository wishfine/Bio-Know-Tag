import json
from pathlib import Path

import pytest

from bio_know_tag.full_adaptive_vote_runner import build_disagreement_inputs, aggregate_adaptive_model, run_full_adaptive_votes


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _prediction(qid: str, labels: list[str], *, context: bool = False) -> dict:
    return {
        "question_id": qid,
        "selected_labels": [{"label_id": label_id} for label_id in labels],
        "context_insufficient": context,
        "need_expand_recall": False,
    }


def test_disagreement_inputs_compare_sets_and_review_flags(tmp_path: Path):
    qids = ["q1", "q2", "q3"]
    units = tmp_path / "units.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(units, [{"question_id": qid, "stem": qid} for qid in qids])
    _write_jsonl(candidates, [{"question_id": qid, "candidates": []} for qid in qids])
    votes = tmp_path / "votes"
    _write_jsonl(votes / "qwen1.jsonl", [
        _prediction("q1", ["A", "B"]), _prediction("q2", ["A"]), _prediction("q3", []),
    ])
    _write_jsonl(votes / "qwen2.jsonl", [
        _prediction("q1", ["B", "A"]), _prediction("q2", ["B"]), _prediction("q3", [], context=True),
    ])
    _write_jsonl(votes / "ds1.jsonl", [
        _prediction("q1", ["A"]), _prediction("q2", []), _prediction("q3", ["C"]),
    ])
    _write_jsonl(votes / "ds2.jsonl", [
        _prediction("q1", ["A"]), _prediction("q2", []), _prediction("q3", ["C"]),
    ])
    paths = {name: votes / f"{name}.jsonl" for name in ("qwen1", "qwen2", "ds1", "ds2")}
    output = tmp_path / "disagreements"

    report = build_disagreement_inputs(units, candidates, paths, output)

    assert report["input"] == 3
    assert report["qwen"]["disagreements"] == 2
    assert report["qwen"]["label_set_disagreements"] == 1
    assert report["qwen"]["review_flag_only_disagreements"] == 1
    assert report["ds"]["disagreements"] == 0
    assert [json.loads(line)["question_id"] for line in (output / "qwen" / "units.jsonl").read_text().splitlines()] == ["q2", "q3"]
    assert (output / "ds" / "units.jsonl").read_text() == ""
    assert build_disagreement_inputs(units, candidates, paths, output) == report

    _write_jsonl(votes / "qwen1.jsonl", [_prediction("q1", ["A"]), _prediction("q2", ["A"]), _prediction("q3", [])])
    with pytest.raises(ValueError, match="source predictions changed"):
        build_disagreement_inputs(units, candidates, paths, output)


def test_adaptive_aggregation_does_not_fabricate_third_vote(tmp_path: Path):
    _write_jsonl(tmp_path / "votes/qwen1/predictions.jsonl", [_prediction("q1", ["A"]), _prediction("q2", ["A", "B"])])
    _write_jsonl(tmp_path / "votes/qwen2/predictions.jsonl", [_prediction("q1", ["A"]), _prediction("q2", ["B", "C"])])
    _write_jsonl(tmp_path / "votes/qwen3_disagreement/predictions.jsonl", [_prediction("q2", ["C"])])
    counts = aggregate_adaptive_model("qwen", tmp_path)
    results = list(map(json.loads, (tmp_path / "model_consensus/qwen.jsonl").read_text().splitlines()))
    assert counts["third_vote_used"] == 1
    assert results[0]["actual_vote_count"] == 2
    assert results[1]["actual_vote_count"] == 3
    assert {row["label_id"] for row in results[1]["selected_labels"]} == {"B", "C"}
    assert all(not row["usable_for_training"] for row in results)


def test_adaptive_controller_four_votes_then_only_disagreement_third(tmp_path: Path, monkeypatch):
    from bio_know_tag.adjudication import PARENT_PROMPT_VERSION, PROMPT_VERSION
    from bio_know_tag.three_vote_runner import _sha256
    units, candidates, labels = (tmp_path / name for name in ("units.jsonl", "candidates.jsonl", "labels.jsonl"))
    _write_jsonl(units, [{"question_id": "q1"}, {"question_id": "q2"}])
    _write_jsonl(candidates, [{"question_id": "q1", "candidates": []}, {"question_id": "q2", "candidates": []}])
    _write_jsonl(labels, [{"label_id": "A"}, {"label_id": "B"}])
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            vote_dir = Path(command[command.index("--run-dir") + 1])
            model = command[command.index("--model") + 1]
            source = Path(command[command.index("--units") + 1])
            qs = [json.loads(line)["question_id"] for line in source.read_text().splitlines()]
            rows = []
            for qid in qs:
                chosen = ["B"] if vote_dir.name == "qwen2" and qid == "q2" else ["A"]
                rows.append({**_prediction(qid, chosen), "model": model})
            _write_jsonl(vote_dir / "predictions.jsonl", rows)
            hashes = {"units": _sha256(source), "candidates": _sha256(Path(command[command.index("--candidates") + 1])), "labels": _sha256(labels)}
            (vote_dir / "report.json").write_text(json.dumps({"success": len(qs), "error": 0, "model": model}))
            config = {"endpoint": command[command.index("--endpoint") + 1], "timeout": float(command[command.index("--timeout") + 1]), "retries": int(command[command.index("--retries") + 1]), "retry_delay": float(command[command.index("--retry-delay") + 1]), "request_interval": 0, "enable_thinking": False}
            (vote_dir / "run_manifest.json").write_text(json.dumps({"input_sha256": hashes, "model": model, "temperature": 0, "stream": True, "max_tokens": 1024, "request_config": config, "prompt_versions_by_unit_type": {"composite_parent_extra": PARENT_PROMPT_VERSION, "other": PROMPT_VERSION}}))
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("bio_know_tag.full_adaptive_vote_runner.subprocess.Popen", Process)
    output = tmp_path / "adaptive"
    result = run_full_adaptive_votes(units, candidates, labels, output, preflight=False)
    assert len(calls) == 5
    assert result["conditional_third_votes"]["qwen"]["disagreements"] == 1
    assert result["conditional_third_votes"]["ds"]["disagreements"] == 0
    assert all(command[command.index("--workers") + 1] == "35" for command in calls)
    assert all("--disable-thinking" in command for command in calls)
    run_full_adaptive_votes(units, candidates, labels, output, preflight=False)
    assert len(calls) == 5
    with pytest.raises(ValueError, match="manifest mismatch"):
        run_full_adaptive_votes(units, candidates, labels, output, preflight=False, workers_per_vote=36)
    child_path = output / "votes/qwen1/run_manifest.json"
    child = json.loads(child_path.read_text())
    child["request_config"]["enable_thinking"] = True
    child_path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match="child request configuration mismatch"):
        run_full_adaptive_votes(units, candidates, labels, output, preflight=False)


def test_seed_preserves_old_evidence_and_refuses_live_controller(tmp_path: Path):
    import fcntl
    from bio_know_tag.adjudication import PARENT_PROMPT_VERSION, PROMPT_VERSION
    from bio_know_tag.full_adaptive_vote_runner import FIRST_VOTES, _seed_from_six_vote
    source = tmp_path / "old"
    output = tmp_path / "new"
    source.mkdir()
    output.mkdir()
    manifest = {
        "input_sha256": {"units": "u", "candidates": "c", "labels": "l"},
        "input_paths": {"units": "u", "candidates": "c", "labels": "l"},
        "retry_delay": 1,
        "thinking_override": {"qwen": False, "ds": False},
        "services": {model: {"endpoint": model, "model": model, "max_tokens": 1024, "timeout": 300, "retries": 5} for model in ("qwen", "ds")},
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    originals = {}
    for vote in FIRST_VOTES:
        path = source / "votes" / vote
        path.mkdir(parents=True)
        model = "qwen" if vote.startswith("qwen") else "ds"
        child = {
            "input_sha256": manifest["input_sha256"], "model": model,
            "input_paths": manifest["input_paths"], "max_tokens": 1024,
            "temperature": 0, "stream": True,
            "request_config": {"endpoint": model, "timeout": 300, "retries": 5, "retry_delay": 1, "request_interval": 0, "enable_thinking": False},
            "prompt_versions_by_unit_type": {"composite_parent_extra": PARENT_PROMPT_VERSION, "other": PROMPT_VERSION},
        }
        (path / "run_manifest.json").write_text(json.dumps(child))
        originals[vote] = b'{"question_id":"q1","error":null}\nunfinished'
        (path / "evidence.jsonl").write_bytes(originals[vote])
    with (source / "controller.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still running"):
            _seed_from_six_vote(source, output, manifest)
    report = _seed_from_six_vote(source, output, manifest)
    for vote in FIRST_VOTES:
        assert (source / "votes" / vote / "evidence.jsonl").read_bytes() == originals[vote]
        assert (output / "votes" / vote / "evidence.jsonl").read_bytes().endswith(b"\n")
        assert list((output / "votes" / vote).glob("evidence.incomplete-tail.*.bin"))
    assert _seed_from_six_vote(source, output, manifest) == report


def test_adaptive_process_failure_terminates_other_votes(tmp_path: Path, monkeypatch):
    from bio_know_tag.full_adaptive_vote_runner import _run_vote_processes
    processes = []

    class Process:
        def __init__(self, command, **kwargs):
            self.vote = Path(command[command.index("--run-dir") + 1]).name
            self.terminated = False
            processes.append(self)
        def poll(self):
            if self.vote == "ds1":
                return 1
            return 0 if self.terminated else None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return self.poll()

    monkeypatch.setattr("bio_know_tag.full_adaptive_vote_runner.subprocess.Popen", Process)
    service = {"endpoint": "http://test", "model": "test", "timeout": 300, "max_tokens": 1024, "retries": 5}
    specs = [(vote, tmp_path / "u", tmp_path / "c", tmp_path / vote, service) for vote in ("qwen1", "ds1")]
    with pytest.raises(RuntimeError, match="incomplete adaptive vote ds1"):
        _run_vote_processes(specs, labels=tmp_path / "l", output=tmp_path, workers=35, retry_delay=1)
    assert processes[0].terminated
