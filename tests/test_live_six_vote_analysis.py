import json
import sys
from pathlib import Path

import pytest

from bio_know_tag.live_six_vote_analysis import analyze_live_six_vote


VOTES = ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3")


def _record(qid: str, labels: list[str], *, error: str | None = None) -> dict:
    return {
        "question_id": qid,
        "error": error,
        "candidate_code_map": {"C01": "L1", "C02": "L2"},
        "parsed_response": {"selected": ["C01" if x == "L1" else "C02" for x in labels]} if error is None else None,
        "finish_reason": "stop" if error is None else "length",
    }


def test_live_six_vote_uses_only_common_successful_questions(tmp_path: Path):
    selections = {
        "qwen1": {"a": ["L1"], "b": ["L1"], "c": [], "d": ["L1"], "e": ["L1"]},
        "qwen2": {"a": ["L1"], "b": ["L1"], "c": [], "d": ["L1"], "e": ["L1"]},
        "qwen3": {"a": ["L1"], "b": ["L2"], "c": [], "e": ["L1"]},
        "ds1": {"a": ["L1"], "b": ["L2"], "c": ["L1"], "d": ["L1"], "e": ["L1"]},
        "ds2": {"a": ["L1"], "b": ["L2"], "c": ["L1"], "d": ["L1"], "e": ["L1"]},
        "ds3": {"a": ["L1"], "b": ["L2"], "c": ["L1"], "d": ["L1"], "e": ["L1"]},
    }
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8") as output:
            for qid, ids in selections[vote].items():
                output.write(json.dumps(_record(qid, ids, error="bad" if vote == "qwen2" and qid == "e" else None)) + "\n")
            output.write('{"question_id":"unfinished"')
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"label_id":"L1","label_name":"甲"}\n{"label_id":"L2","label_name":"乙"}\n', encoding="utf-8")

    report = analyze_live_six_vote(tmp_path, labels_path=labels)

    assert report["all_six_common_questions"] == 3
    assert report["votes"]["qwen3"]["successful_questions"] == 4
    assert report["votes"]["qwen2"]["errors"] == 1
    assert report["within_model"]["qwen"]["all_same"] == 2
    assert report["within_model"]["qwen"]["two_same_one_different"] == 1
    assert report["within_model"]["qwen"]["first_two_same_but_third_differs"] == 1
    assert report["within_model"]["ds"]["all_same"] == 3
    assert report["cross_model"]["majority_same"] == 1
    assert report["cross_model"]["majority_different"] == 2
    assert report["cross_model"]["both_unanimous_different"] == 1
    assert {item["question_id"] for item in report["samples"]["stable_cross_model_difference"]} == {"c"}
    by_label = {row["label_id"]: row for row in report["labels"]}
    assert by_label["L1"]["qwen_unstable"] == 1
    assert by_label["L1"]["qwen_majority_only"] == 1
    assert by_label["L2"]["ds_majority_only"] == 1


def test_live_six_vote_latest_success_survives_retried_error(tmp_path: Path):
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8") as output:
            output.write(json.dumps(_record("a", ["L1"])) + "\n")
            if vote == "qwen1":
                output.write(json.dumps(_record("a", [], error="failed retry")) + "\n")
    report = analyze_live_six_vote(tmp_path)
    assert report["all_six_common_questions"] == 1
    assert report["within_model"]["qwen"]["all_same"] == 1
    assert report["votes"]["qwen1"]["evidence_rows"] == 2
    assert report["votes"]["qwen1"]["errors"] == 1


def test_live_six_vote_shows_when_third_vote_resolves_first_two_split(tmp_path: Path):
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        chosen = ["L2"] if vote == "qwen2" else ["L1"]
        path.write_text(json.dumps(_record("a", chosen)) + "\n", encoding="utf-8")
    report = analyze_live_six_vote(tmp_path)
    qwen = report["within_model"]["qwen"]
    assert qwen["first_two_different"] == 1
    assert qwen["third_agrees_first_when_split"] == 1
    assert qwen["third_agrees_second_when_split"] == 0
    assert qwen["third_differs_both_when_split"] == 0


def test_live_six_vote_rejects_mismatched_input_manifests(tmp_path: Path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"input_sha256": {"units": "frozen-units"}}), encoding="utf-8",
    )
    for vote in VOTES:
        directory = tmp_path / "votes" / vote
        directory.mkdir(parents=True)
        (directory / "evidence.jsonl").write_text(json.dumps(_record("a", ["L1"])) + "\n")
        (directory / "run_manifest.json").write_text(json.dumps({
            "input_sha256": {"units": "different" if vote == "ds3" else "frozen-units"}
        }))
    with pytest.raises(ValueError, match="ds3.*units"):
        analyze_live_six_vote(tmp_path)


def test_live_six_vote_accepts_full_runner_no_thinking_manifests(tmp_path: Path):
    parent = {
        "input_sha256": {"units": "u", "candidates": "c", "labels": "l"},
        "thinking_override": {"qwen": False, "ds": False},
        "services": {"qwen": {"model": "qwen"}, "ds": {"model": "ds"}},
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(parent), encoding="utf-8")
    for vote in VOTES:
        directory = tmp_path / "votes" / vote
        directory.mkdir(parents=True)
        (directory / "evidence.jsonl").write_text(json.dumps(_record("a", ["L1"])) + "\n")
        (directory / "run_manifest.json").write_text(json.dumps({
            "input_sha256": parent["input_sha256"],
            "model": "qwen" if vote.startswith("qwen") else "ds",
            "request_config": {"enable_thinking": False},
        }))
    report = analyze_live_six_vote(tmp_path)
    assert report["manifest_alignment"]["vote_manifests_checked"] == 6


def test_live_six_vote_empty_intersection_renders_report(tmp_path: Path):
    from scripts.analyze_live_full_six_votes import _markdown

    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_record(vote, ["L1"])) + "\n", encoding="utf-8")
    report = analyze_live_six_vote(tmp_path)
    assert report["all_six_common_questions"] == 0
    assert "0 道题" in _markdown(report)


def test_live_six_vote_refuses_snapshot_larger_than_memory_cap(tmp_path: Path):
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_record("a", ["L1"])) + "\n"
            + json.dumps(_record("b", ["L1"])) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="memory cap"):
        analyze_live_six_vote(tmp_path, max_success_per_vote=1)


def test_live_six_vote_cli_refuses_output_inside_source_tree(tmp_path: Path, monkeypatch):
    from scripts.analyze_live_full_six_votes import main

    source = tmp_path / "adjudication-no-thinking"
    source.mkdir()
    output = source / "votes" / "qwen1"
    monkeypatch.setattr(sys, "argv", [
        "analyze_live_full_six_votes.py", "--votes-root", str(source),
        "--run-dir", str(output),
    ])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert not (output / "report.json").exists()
