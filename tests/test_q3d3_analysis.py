import json
import hashlib
import sys
from pathlib import Path

import pytest

from bio_know_tag.q3d3_analysis import analyze_q3d3


VOTES = ("qwen1", "qwen2", "qwen3", "ds1", "ds2", "ds3")


def _row(qid: str, selected: list[str]) -> dict:
    codes = {"L1": "C01", "L2": "C02", "L3": "C03"}
    return {
        "question_id": qid,
        "candidate_code_map": {code: label for label, code in codes.items()},
        "parsed_response": {"selected": [codes[label] for label in selected]},
        "error": None,
    }


def test_q3d3_counts_labels_questions_and_other_label_strength(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"label_id": x, "label_name": x}) + "\n" for x in ("L1", "L2", "L3")))
    units = tmp_path / "units.jsonl"
    units.write_text("".join(json.dumps({"question_id": qid, "stem": f"题干{qid}", "unit_type": "standalone"}, ensure_ascii=False) + "\n" for qid in ("q1", "q2", "q3")))
    per_vote = {
        "qwen1": {"q1": ["L1", "L2"], "q2": ["L2"], "q3": ["L1", "L3"]},
        "qwen2": {"q1": ["L1", "L2"], "q2": ["L2"], "q3": ["L1", "L3"]},
        "qwen3": {"q1": ["L1", "L2"], "q2": ["L2"], "q3": ["L1", "L3"]},
        "ds1": {"q1": ["L1", "L2"], "q2": ["L2"], "q3": ["L1", "L3"]},
        "ds2": {"q1": ["L1", "L2"], "q2": ["L2"], "q3": ["L1", "L3"]},
        "ds3": {"q1": ["L1"], "q2": ["L2"], "q3": ["L1", "L3"]},
    }
    for vote in VOTES:
        path = tmp_path / "votes" / vote / "evidence.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(_row(qid, selected)) + "\n" for qid, selected in per_vote[vote].items()))

    report = analyze_q3d3(tmp_path, labels_path=labels, units_path=units, top_k=3, sample_per_label=3)

    assert report["all_six_common_questions"] == 3
    assert report["q3d3_label_assignments"] == 4
    assert report["questions_with_q3d3"] == 3
    assert report["questions_with_only_q3d3_among_ever_selected"] == 2
    assert report["other_label_vote_pairs"]["Q3/D2"] == 1
    by_label = {row["label_id"]: row for row in report["labels"]}
    assert by_label["L1"]["q3d3_count"] == 2
    assert by_label["L1"]["q3d3_share"] == 0.5
    assert by_label["L1"]["questions_with_other_near_consensus"] == 1
    assert by_label["L1"]["questions_with_other_q3d3"] == 1
    assert any(item["stem"] == "题干q1" for item in report["samples"]["L1"])
    from scripts.analyze_q3d3_live import _markdown
    assert "Q3/D2" in _markdown(report)


def test_q3d3_refuses_question_text_from_different_units_file(tmp_path: Path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"label_id":"L1","label_name":"甲"}\n')
    units = tmp_path / "units.jsonl"
    units.write_text('{"question_id":"q1","stem":"wrong text"}\n')
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "input_sha256": {"units": hashlib.sha256(b'{"question_id":"q1","stem":"correct text"}\n').hexdigest()}
    }))
    for vote in VOTES:
        directory = tmp_path / "votes" / vote
        directory.mkdir(parents=True)
        directory.joinpath("evidence.jsonl").write_text(json.dumps({
            "question_id": "q1", "candidate_code_map": {"C01": "L1"},
            "parsed_response": {"selected": ["C01"]}, "error": None,
        }) + "\n")
        directory.joinpath("run_manifest.json").write_text(json.dumps({
            "input_sha256": json.loads((tmp_path / "run_manifest.json").read_text())["input_sha256"]
        }))
    with pytest.raises(ValueError, match="units.*SHA256"):
        analyze_q3d3(tmp_path, labels_path=labels, units_path=units)


def test_q3d3_cli_refuses_occupied_output_dir(tmp_path: Path, monkeypatch):
    from scripts.analyze_q3d3_live import main

    source = tmp_path / "votes-source"
    live_file = source / "votes" / "qwen1" / "evidence.jsonl"
    live_file.parent.mkdir(parents=True)
    live_file.write_text("protected evidence\n")
    output = tmp_path / "analysis-output"
    output.mkdir()
    (output / "report.json").symlink_to(live_file)
    monkeypatch.setattr(sys, "argv", [
        "analyze_q3d3_live.py", "--votes-root", str(source),
        "--units", str(tmp_path / "units.jsonl"), "--run-dir", str(output),
    ])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert live_file.read_text() == "protected evidence\n"
